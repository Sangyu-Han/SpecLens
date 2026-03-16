# src/sae_index/decile_parquet_ledger.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
from .registry_utils import ensure_dir, unique_basename
import torch

def _part(sample_id: int, M: int = 128) -> int:
    return int(sample_id) % int(M)

DECILES_SCHEMA = pa.schema([
    ("run_id", pa.string()),            # 실험/인덱싱 런 식별자(필수 권장)
    ("layer", pa.string()),             # 레이어 이름
    ("unit", pa.int32()),               # SAE feature index
    ("score", pa.float32()),            # 활성값 혹은 SAE 출력 점수
    ("decile", pa.int32()),             # 0..(num_deciles-1)
    ("rank_in_decile", pa.int32()),     # decile 내 상위 정렬 순위(0=1등)
    ("sample_id", pa.int64()),          # 원본 샘플 ID
    ("frame_idx", pa.int32()),          # 토큰/좌표가 속한 프레임 인덱스
    ("y", pa.int32()),                  # 좌표(-1은 토큰류 의미)
    ("x", pa.int32()),
    ("prompt_id", pa.int64()),          # 프롬프트 세트 ID(주입 프레임 폴백 포함)
    ("uid", pa.int64()),                # (옵션) 객체 UID(t0 기준). 없으면 -1
    ("stride_step", pa.int32()),        # stride 정보
    ("meta_json", pa.string()),         # (옵션) 부가 메타 JSON 문자열
    # 파티션 컬럼은 write 시 append
])

class DecileParquetLedger:
    """
    deciles/ 이하에 파케셋 저장.
    파티션: layer(string)/decile(int)/part(int=sample_id%M)
    """
    def __init__(self, root_dir: str | Path, *, M_part: int = 128, compression: str = "zstd"):
        self.root = Path(root_dir)
        self.dir  = self.root / "deciles"
        ensure_dir(self.dir)
        self.M = int(M_part)
        self._pq_kwargs = {
            "compression": compression,
            "use_dictionary": ["layer", "meta_json", "run_id"],
            "write_statistics": True,
        }

    def _schema_with_partitions(self) -> pa.Schema:
        return (
            DECILES_SCHEMA
            .append(pa.field("layer_part", pa.string()))
            .append(pa.field("decile_part", pa.int32()))
            .append(pa.field("part", pa.int32()))
        )

    def write_rows(self, rows: List[Dict[str, Any]], *, rank: int = 0) -> int:
        if not rows:
            return 0

        # (1) 스키마 + 파티션 컬럼 주입
        sch = self._schema_with_partitions()
        for r in rows:
            r["layer_part"]  = str(r["layer"])
            r["decile_part"] = int(r["decile"])
            r["part"]        = _part(int(r["sample_id"]), self.M)
            if "uid" not in r:
                r["uid"] = -1
            if "meta_json" not in r:
                r["meta_json"] = ""

        # (2) Arrow Table 생성
        cols = {}
        for f in sch:
            name = f.name
            vals = [rr.get(name, None) for rr in rows]
            cols[name] = pa.array(vals, type=f.type)
        tbl = pa.table(cols, schema=sch)

        # (3) dataset.write_dataset 로 교체 (+ max_partitions 늘리기)
        base = unique_basename("deciles", rank=rank)

        # 파티션 정의(Hive 스타일)
        part_schema = pa.schema([
            ("layer_part", pa.string()),
            ("decile_part", pa.int32()),
            ("part", pa.int32()),
        ])
        partitioning = ds.partitioning(part_schema, flavor="hive")

        # Parquet write 옵션 구성 (기존 _pq_kwargs 재사용)
        fmt = ds.ParquetFileFormat()
        file_options = fmt.make_write_options(**self._pq_kwargs)

        # 파티션 상한(기본 1024)을 충분히 키움
        max_parts = max(8192, self.M * 16)  # 여유 있게

        ds.write_dataset(
            data=tbl,
            base_dir=str(self.dir),
            format="parquet",
            partitioning=partitioning,
            basename_template=f"{base}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            file_options=file_options,
            max_partitions=max_parts,
            use_threads=True,
        )
        return len(rows)


    # 조회 유틸
    def as_dataset(self):
        return ds.dataset(str(self.dir), format="parquet", partitioning="hive")
    def topn_for(self, *, layer: str, unit: int, decile: int, n: int) -> pa.Table:
        dset = self.as_dataset()
        f = (
            (ds.field("layer_part") == str(layer)) &      # 파티션 프루닝
            (ds.field("decile_part") == int(decile)) &    # 파티션 프루닝
            (ds.field("unit") == int(unit)) &
            (ds.field("rank_in_decile") < int(n))         # 상위 n만
        )
        # 필요한 컬럼만 최소화 (정말 필요한 것만 남기면 더 빨라짐)
        cols = ["layer","unit","score","decile","rank_in_decile",
                "sample_id","frame_idx","y","x","prompt_id","uid","stride_step","run_id"]
        tbl = dset.to_table(filter=f, columns=cols)

        # rank 기준으로 정렬(필요 시). 이미 n개 수준이라 비용 매우 작음.
        if tbl.num_rows <= 1:
            return tbl
        idx = pc.sort_indices(tbl, sort_keys=[("rank_in_decile", "ascending")])
        return pc.take(tbl, idx)

    def units_for_layer(self, layer: str) -> List[int]:
        """
        Return sorted list of SAE unit indices that have at least one row for the given layer.
        """
        try:
            dset = self.as_dataset()
        except (FileNotFoundError, pa.ArrowInvalid):
            return []

        tbl = dset.to_table(
            filter=ds.field("layer_part") == str(layer),
            columns=["unit"],
        )
        if tbl.num_rows == 0:
            return []
        return sorted(pc.unique(tbl["unit"]).to_pylist())

    def __repr__(self) -> str:
        compression = self._pq_kwargs.get("compression")
        return (
            f"DecileParquetLedger(root='{self.root}', dir='{self.dir}', "
            f"M={self.M}, compression={compression!r})"
        )

    # def topn_for(self, *, layer: str, unit: int, decile: int, n: int) -> pa.Table:
    #     dset = self.as_dataset()
    #     f = (
    #         (ds.field("layer") == str(layer)) &
    #         (ds.field("unit") == int(unit)) &
    #         (ds.field("decile") == int(decile))
    #     )
    #     tbl = dset.to_table(filter=f, columns=[
    #         "layer","unit","score","decile","rank_in_decile",
    #         "sample_id","frame_idx","y","x","prompt_id","uid","stride_step","run_id"
    #     ])
    #     # score 내림차순 정렬 후 상위 n
    #     if tbl.num_rows <= n:
    #         return tbl
    #     import pyarrow.compute as pc
    #     idx = pc.sort_indices(tbl, sort_keys=[("score", "descending")])
    #     tbl_sorted = pc.take(tbl, idx)
    #     return tbl_sorted.slice(0, n)
