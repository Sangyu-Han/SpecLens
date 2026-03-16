"""Reusable SAE training runner shared across packs."""
import os

# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")           # 에러 위치 정확히 찍게
# os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1048576")  # FlightRecorder
# os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
# os.environ.setdefault("NCCL_DEBUG", "WARN")
# os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")      # NCCL이 조기 실패 전파
# os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")  # 디버그 메시지 자세히
# os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")   # C++ 레벨 INFO 이상
# 단일 노드면 IB 끄면 깔끔한 경우 많음(InfiniBand 안 쓰면 권장)
# os.environ.setdefault("NCCL_IB_DISABLE", "1")
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import re  # <= 파일명에서 step 파싱용
import gc

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from src.core.indexing.registry_utils import load_obj
from src.core.sae.kmeans.utils import sanitize_layer_name, load_centroids, centroids_path
# project root
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ✅ SAM2 리포 루트를 그대로 올리면 'training'과 'sam2'가 둘 다 import 가능
sam2_root = project_root / "third_party" / "sam2_src"
if str(sam2_root) not in sys.path:
    sys.path.insert(0, str(sam2_root))


from src.core.sae.activation_stores.universal_activation_store import (
    UniversalActivationStore,
)

from src.core.sae.registry import create_sae, list_available_saes

import time
try:
    import wandb
except Exception:
    wandb = None
    
logger = logging.getLogger(__name__)


class CosineScheduler:
    """Cosine LR scheduler with linear warmup (matches overcomplete official)."""

    def __init__(self, optimizer, base_value, final_value, total_iters,
                 warmup_iters=0, start_warmup_value=0.0):
        import numpy as np
        self.optimizer = optimizer
        self.final_value = float(final_value)
        self.total_iters = int(total_iters)

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        cosine = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / max(len(iters), 1))
        )
        self.schedule = torch.tensor(
            np.concatenate((warmup_schedule, cosine)), dtype=torch.float32
        )
        self.iter = 0

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        return float(self.schedule[it])

    def step(self):
        self.iter += 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self[self.iter]


class SAETrainingPipeline:
    def __init__(self, config_path: str, rank: int, world_size: int):
        self.config = self._load_config(config_path)
        self.pack_cfg = self.config.get("pack") or {}
        self._apply_pack_setup()
        self.rank = rank
        self.world_size = world_size
        # ✅ torchrun이 주는 LOCAL_RANK 사용 (노드 내 GPU 인덱스)
        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.config.setdefault("model", {})["device"] = str(self.device)

        # (선택) 매핑 로그
        logger.info(f"[DDP] rank={self.rank}, local_rank={self.local_rank}, world={self.world_size}, device={self.device}, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

        self._setup_logging()
        self._setup_output_dirs()

        self.model: nn.Module | None = None
        self.dataset = None
        self.sampler: DistributedSampler | None = None
        self.activation_store: UniversalActivationStore | None = None
        self.sae_models: dict[str, nn.Module] = {}
        self.my_layers_to_train: list[str] = []
        self.layer_owners: Dict[str, int] = {}  # <= 파이프라인에도 저장
        # wandb 관련 설정
        self.wandb_run = None
        self.wandb_cfg = self.config.get("logging", {}).get("wandb", {})
        self.tokens_seen: Dict[str, int] = {}   # ✅ 레이어 토큰 카운터
        # validation 캐시
        self.validation_cache: dict[str, torch.Tensor] = {}
        self.validation_ready: bool = False
        self.validation_cfg = self.config.get("validation", {}) or {}

    def _apply_pack_setup(self) -> None:
        pack_cfg = self.pack_cfg
        sys_paths = list(pack_cfg.get("sys_paths", []))
        for rel in reversed(sys_paths):
            path = Path(rel)
            if not path.is_absolute():
                path = project_root / path
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        setup_target = pack_cfg.get("setup")
        if setup_target:
            setup_fn = load_obj(setup_target)
            options = pack_cfg.get("options", pack_cfg)
            try:
                setup_fn(options, project_root=project_root)
            except TypeError:
                setup_fn(options)


    # ------------------------------- utils ---------------------------------
    def _fmt_num(self, n: int) -> str:
        # 가독성 높은 단위 표기 (k/M)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}k"
        return f"{n}"

    def _print_buffer_progress(self, force: bool = False):
        """각 레이어 큐의 '남은 토큰 / 초기 토큰' 상황을 tqdm처럼 출력."""
        # 너무 자주 찍지 않도록 스로틀
        if not hasattr(self, "_last_buf_print"):
            self._last_buf_print = 0.0
        now = time.time()
        interval = float(self.wandb_cfg.get("buffer_print_interval_sec", 5.0))
        if not force and (now - self._last_buf_print < interval):
            return
        self._last_buf_print = now

        if self.activation_store is None:
            return

        stats = self.activation_store.get_buffer_progress()
        if not stats:
            return

        lines = []
        width = 24
        for lname, st in sorted(stats.items()):
            cur  = st["current"]
            init = st["init"]
            peak = st["peak"]
            low  = st["low"]
            high = st["high"]

            # '처음 채워진 토큰수'가 0이면 fallback으로 peak 또는 low 사용
            base = init if init > 0 else (peak if peak > 0 else max(low, 1))
            ratio = min(1.0, cur / float(base)) if base > 0 else 0.0
            filled = int(ratio * width)
            bar = "█" * filled + "·" * (width - filled)

            # 이 랭크가 오너인 레이어만 강조(별표) — 출력 난잡함 줄이기
            owner = self.layer_owners.get(lname, -1)
            star = "*" if owner == self.rank else " "

            lines.append(
                f"[r{self.rank}{star}] {lname:<44} "
                f"{self._fmt_num(cur):>6}/{self._fmt_num(base):<6} "
                f"|low {self._fmt_num(low):>5} high {self._fmt_num(high):>5}| "
                f"{bar} {ratio*100:5.1f}%"
            )

        # 랭크별로 섞여도 읽기 쉽게 블록 단위로 출력
        print("\n".join(lines) + "\n", flush=True)

    def _log_gpu_memory(self, tag: str) -> None:
        """Log CUDA memory usage (allocated/reserved/peak) for diagnostics."""
        if not torch.cuda.is_available():
            return
        try:
            dev = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(dev) / 1e9
            reserved = torch.cuda.memory_reserved(dev) / 1e9
            peak = torch.cuda.max_memory_allocated(dev) / 1e9
            # CPU memory (RSS + shmem)
            import os as _os
            pid = _os.getpid()
            rss_mb = shmem_mb = 0.0
            try:
                with open(f"/proc/{pid}/status") as _f:
                    for _line in _f:
                        if _line.startswith("VmRSS:"):
                            rss_mb = int(_line.split()[1]) / 1024
                        elif _line.startswith("RssShmem:"):
                            shmem_mb = int(_line.split()[1]) / 1024
            except Exception:
                pass
            logger.info(
                f"[mem] {tag} GPU alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB | "
                f"CPU RSS={rss_mb:.0f}MB shmem={shmem_mb:.0f}MB"
            )
        except Exception:
            pass
    def _barrier(self):
        if dist.is_initialized():
            if dist.get_backend() == "nccl":
                # ✅ 로컬 디바이스를 명시해서 경고 제거
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()
    # ------------------------------- W&B ---------------------------------
    def _setup_wandb(self):
        if not self.wandb_cfg or not self.wandb_cfg.get("enabled", False):
            return
        if wandb is None:
            logger.warning("wandb가 설치되어 있지 않습니다. pip install wandb 로 설치하세요.")
            return
        mode = self.wandb_cfg.get("mode", "online")
        if mode == "disabled":
            return

        log_all_ranks = bool(self.wandb_cfg.get("log_all_ranks", True))
        if not log_all_ranks and self.rank != 0:
            logger.info("[Rank %d] wandb logging disabled (rank>0).", self.rank)
            return

        name = f'{self.wandb_cfg.get("run_name","sam2_sae")}-r{self.rank}'
        try:
            self.wandb_run = wandb.init(
                project=self.wandb_cfg.get("project", "sam2-sae"),
                entity=self.wandb_cfg.get("entity"),
                name=name,
                group=self.wandb_cfg.get("group"),
                tags=self.wandb_cfg.get("tags"),
                mode=mode,
                config=self.config,
                settings=wandb.Settings(
                    start_method="thread",
                    _service_wait=int(self.wandb_cfg.get("service_wait", 120)),
                    _disable_meta=True,
                ),
            )
        except Exception as exc:
            logger.warning("wandb 초기화에 실패했습니다 (rank=%d): %s", self.rank, exc)
            self.wandb_run = None
            self.wandb_cfg["enabled"] = False
            return

        if self.wandb_run:
            self.wandb_run.define_metric("global/step")
            self.wandb_run.define_metric("global/*", step_metric="global/step")

    def _wandb_log(self, data: dict, *, commit: bool | None = None) -> None:
        if not self.wandb_run:
            return
        try:
            if commit is None:
                wandb.log(data)
            else:
                wandb.log(data, commit=commit)
        except Exception as exc:
            logger.warning("wandb.log failed (rank=%d): %s", self.rank, exc)
            self.wandb_run = None
            self.wandb_cfg["enabled"] = False

    
    def _assign_layer_owners(self):
        """discover 이후 한 번만 호출. store와 pipeline에 동일 매핑 주입"""
        layers = self.activation_store.expanded_hook_points
        self.alllayers_to_train = layers  # pipeline에 저장용
        # store가 이미 owners를 세팅한 경우(팩별 커스텀) 우선 사용
        pre = getattr(self.activation_store, "layer_owners", {}) or {}
        if pre:
            owners = dict(pre)
            # 누락된 레이어는 round-robin으로 보완
            for i, ln in enumerate(layers):
                owners.setdefault(ln, i % self.world_size)
            self.layer_owners = owners
            self.activation_store.set_layer_owners(owners)
            logger.info(
                f"[Rank {self.rank}] owner map loaded from store (world_size={self.world_size})"
            )
            return

        # 기본: round-robin
        owners = {ln: (i % self.world_size) for i, ln in enumerate(layers)}
        self.layer_owners = owners
        # 스토어에도 1회 세팅
        self.activation_store.set_layer_owners(owners)
        logger.info(f"[Rank {self.rank}] owner map set (world_size={self.world_size})")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError(f"Config is empty or invalid YAML: {config_path}")
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a dict, got {type(cfg)} from {config_path}")
        return cfg

    def _setup_logging(self):
        lvl = getattr(logging, self.config.get("logging", {}).get("level", "INFO"))
        log_dir = Path(self.config.get("logging", {}).get("log_dir", "outputs/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        # 랭크별 파일
        log_file = log_dir / f"train_sae_r{self.rank}.log"
        # 중복 핸들러 방지
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def _setup_output_dirs(self):
        save_path = self.config["sae"]["output"]["save_path"]
        os.makedirs(save_path, exist_ok=True)
        
    def _sanitize_layer_dir(self, lname: str) -> Path:
        """체크포인트 저장 디렉토리(레이어 이름을 파일시스템 세이프하게)."""
        save_root = Path(self.config["sae"]["output"]["save_path"])
        return save_root / sanitize_layer_name(lname)

    def _parse_step_from_path(self, p: Path) -> int:
        """파일명에서 step 숫자 파싱 (없으면 -1). 예: step_0000123_tokens_....pt"""
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    def _find_latest_ckpt_path(self, lname: str) -> Path | None:
        """레이어 디렉토리에서 가장 step이 큰 체크포인트 경로 반환."""
        layer_dir = self._sanitize_layer_dir(lname)
        if not layer_dir.exists():
            return None
        candidates = list(layer_dir.glob("*.pt"))
        if not candidates:
            return None
        # step 기준 정렬 (동률이면 수정시간 최신 우선)
        candidates.sort(key=lambda x: (self._parse_step_from_path(x), x.stat().st_mtime))
        return candidates[-1]

    def _move_optimizer_state_to_device(self, opt: torch.optim.Optimizer, device: torch.device):
        """옵티마이저 state 텐서들을 현재 디바이스로 이동 (CPU로 로드된 경우 대비)."""
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def _resume_from_checkpoints(self, opt_for: dict[str, torch.optim.Optimizer],
                                 steps_done: dict[str, int]) -> None:
        """
        각 SAE 레이어별로 가장 step이 큰 체크포인트를 찾아 로드.
        - SAE 가중치/옵티마이저 state/step/tokens_seen 복구
        - W&B를 쓰면 초기 스텝/토큰 카운터도 업데이트
        """
        for lname in self.my_layers_to_train:
            latest = self._find_latest_ckpt_path(lname)
            if latest is None:
                continue

            try:
                pkg = torch.load(latest, map_location="cpu")
            except Exception as e:
                logger.warning(f"[Rank {self.rank}] Failed to load ckpt for {lname} ({latest}): {e}")
                continue

            # 1) SAE 가중치
            sae = self.sae_models[lname]
            try:
                sae.load_state_dict(pkg.get("sae_state", {}), strict=False)
                sae.to(self.device)
            except Exception as e:
                logger.warning(f"[Rank {self.rank}] SAE state load mismatch for {lname}: {e}")

            # 2) 옵티마이저
            opt = opt_for[lname]
            try:
                opt.load_state_dict(pkg.get("optimizer_state", {}))
                self._move_optimizer_state_to_device(opt, self.device)
            except Exception as e:
                logger.warning(f"[Rank {self.rank}] Optimizer state load failed for {lname}: {e}")

            # 3) step / tokens_seen 복구
            restored_step = int(pkg.get("step", self._parse_step_from_path(latest)))
            restored_tokens = int(pkg.get("tokens_seen", 0))
            steps_done[lname] = max(steps_done.get(lname, 0), restored_step)
            self.tokens_seen[lname] = max(self.tokens_seen.get(lname, 0), restored_tokens)

            logger.info(
                f"[Rank {self.rank}] Resumed {lname} from {latest.name} "
                f"(step={restored_step}, tokens_seen={self._fmt_num(restored_tokens)})"
            )

            # Re-init b_norm from centroid mean if loaded as zeros (old checkpoint)
            if hasattr(sae, "b_norm") and sae.b_norm.abs().max().item() < 1e-10:
                act_size = sae.b_norm.shape[0]
                points = self._load_kmeans_centers(lname, act_size)
                if points is not None:
                    centroid_mean = points.mean(dim=0)
                    sae.b_norm.data.copy_(centroid_mean.to(self.device))
                    logger.info(f"[Rank {self.rank}] Re-initialized b_norm from centroid mean for {lname} (old ckpt)")

            # 4) (선택) W&B 초기 카운터 업데이트
            if self.wandb_run:
                safe = lname.replace("/", "_")
                self._wandb_log({f"{safe}/step": steps_done[lname], f"{safe}/token_seen": self.tokens_seen[lname]})
    # ----------------------------- model -----------------------------------

    def _load_model(self) -> nn.Module:
        cfg_m = self.config["model"]
        loader_path = cfg_m.get("loader")
        if not loader_path:
            raise KeyError("config['model']['loader'] must be provided for model construction")

        loader = load_obj(loader_path)
        model = loader(
            model_cfg=cfg_m,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            full_config=self.config,
        )
        model = model.eval()  # SAE 학습 중에는 평가 모드로 고정
        if isinstance(model, DDP):
            return model
        if not isinstance(model, nn.Module):
            raise TypeError(f"Model loader '{loader_path}' must return an nn.Module or DDP instance.")

        if self.world_size <= 1 or not dist.is_initialized():
            if self.world_size > 1 and not dist.is_initialized():
                logger.warning(
                    "[DDP] world_size=%d but process group not initialized; running without DDP. "
                    "If you intended multi-GPU, launch with torchrun to init_process_group.",
                    self.world_size,
                )
            return model

        find_unused = bool(cfg_m.get("find_unused_parameters", False))
        device_ids = [self.local_rank] if self.device.type == "cuda" else None
        output_device = self.local_rank if self.device.type == "cuda" else None
        return DDP(model, device_ids=device_ids, output_device=output_device, find_unused_parameters=find_unused)

    # ----------------------------- dataset ---------------------------------

    def _load_dataset(self):
        dcfg = self.config["dataset"]
        builder_path = dcfg.get("builder")
        if not builder_path:
            raise KeyError("config['dataset']['builder'] must be provided")

        builder = load_obj(builder_path)
        result = builder(
            dataset_cfg=dcfg,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            full_config=self.config,
        )

        dataset = collate_fn = sampler = None
        if isinstance(result, dict):
            dataset = result.get("dataset")
            collate_fn = result.get("collate_fn")
            sampler = result.get("sampler")
        elif isinstance(result, (list, tuple)):
            if len(result) >= 2:
                dataset, collate_fn = result[:2]
            if len(result) >= 3:
                sampler = result[2]
        else:
            raise TypeError(f"Dataset builder '{builder_path}' returned unsupported type: {type(result)}")

        if dataset is None or collate_fn is None:
            raise RuntimeError(f"Dataset builder '{builder_path}' must return both dataset and collate_fn.")

        self.dataset = dataset
        self.collate_fn = collate_fn
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
        if self.rank == 0:
            logger.info(f"[Rank {self.rank}] Dataset ready (builder={builder_path}).")


    # ------------------------- activation store -----------------------------

    def _create_activation_store(self):
        tr = self.config["sae"]["training"]
        # store_config는 UniversalActivationStore가 그대로 읽는 구조
        store_config = {
            "hook_points": self.config["sae"]["layers"],
            "model_batch_size": int(self.config["dataset"]["batch_size"]),
            "num_batches_in_buffer": int(tr.get("n_batches_in_buffer", 4)),
            "device": str(self.device),
            "batch_size": int(tr.get("activation_batch_size", 4096)),
            "buffer_on_cpu": bool(tr.get("buffer_on_cpu", True)),
            "num_workers": int(tr.get("num_workers", self.config.get("dataset", {}).get("num_workers", 0))),
            "discovery": {"batch_size": 1, "override_num_frames": 2, "amp_bfloat16": True},

            # collector/defaults/per_layer
            "collector": {
                "max_collect_batches_per_round": int(tr.get("batches_per_collect_round", 2)),
                "low_watermark_tokens": int(tr.get("low_watermark_tokens", tr.get("activation_batch_size", 4096))),
                "high_watermark_tokens": int(tr.get("high_watermark_tokens", 1_000_000)),
            },
            "defaults": {
                "batch_size": int(tr.get("activation_batch_size", 4096)),
                "low": int(tr.get("low_watermark_tokens", tr.get("activation_batch_size", 4096))),
                "high": int(tr.get("high_watermark_tokens", 1_000_000)),
                "stride": int(tr.get("stride", 1)),
            },
            "per_layer": tr.get("per_layer", {}),

            # 새 섹션들
            "sync": tr.get("sync", {}),   # mode, collect_policy, dynamic_chunking, chunk_mb, ...
            "queue": tr.get("queue", {}), # block_size_tokens, spill_to_disk, ...
            "training": {
                "adaptive_collect": tr.get("adaptive_collect", True),
                "min_collect_batches_per_round": tr.get("min_collect_batches_per_round", 1),
                "max_collect_batches_per_round": tr.get("max_collect_batches_per_round", 4),
            },
        }
        
        store_section = self.config["sae"].get("store") or {}
        factory_path = store_section.get("factory") or self.config["sae"].get("store_factory")
        if not factory_path:
            raise KeyError("config['sae']['store']['factory'] (or 'store_factory') must be provided")
        if self.collate_fn is None:
            raise RuntimeError("collate_fn must be available before activation store creation.")

        store_factory = load_obj(factory_path)
        factory_kwargs = dict(store_section.get("factory_kwargs") or {})
        self.activation_store = store_factory(
            model=self.model,
            cfg=store_config,
            dataset=self.dataset,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            **factory_kwargs,
        )

        # Early hook-spec validation (after model load + dataset ready)
        vcfg = self.config.get("sae", {}).get("validate_hook_points", True)
        if isinstance(vcfg, dict):
            enabled = bool(vcfg.get("enabled", True))
            strict = bool(vcfg.get("strict", True))
            log_possible = bool(vcfg.get("log_possible", True))
            max_list = int(vcfg.get("max_list", 200))
        else:
            enabled = bool(vcfg)
            strict = bool(self.config.get("sae", {}).get("validate_hook_points_strict", True))
            log_possible = bool(self.config.get("sae", {}).get("validate_hook_points_log_possible", True))
            max_list = int(self.config.get("sae", {}).get("validate_hook_points_max_list", 200))

        # Optional: dump possible (expanded) hook specs to a file
        dump_cfg = self.config.get("sae", {}).get("dump_hook_points")
        if isinstance(dump_cfg, dict) and dump_cfg.get("enabled") is False:
            dump_cfg = None
        elif dump_cfg is False:
            dump_cfg = None
        elif dump_cfg is None or dump_cfg is True:
            dump_cfg = {"path": "outputs/hook_specs.txt"}

        if dump_cfg and self.activation_store is not None:
            if isinstance(dump_cfg, dict):
                dump_path = dump_cfg.get("path")
                with_sizes = bool(dump_cfg.get("with_sizes", True))
                max_items = dump_cfg.get("max_items")
                scope = str(dump_cfg.get("scope", "hook_points")).lower()
                include_prefixes = dump_cfg.get("include_prefixes")
                exclude_prefixes = dump_cfg.get("exclude_prefixes")
                max_modules = dump_cfg.get("max_modules")
                tmp_bs = int(dump_cfg.get("tmp_bs", 1))
                tmp_nf = int(dump_cfg.get("tmp_nf", 2))
                use_amp = bool(dump_cfg.get("use_amp", True))
            else:
                dump_path = dump_cfg
                with_sizes = True
                max_items = None
                scope = "hook_points"
                include_prefixes = None
                exclude_prefixes = None
                max_modules = None
                tmp_bs = 1
                tmp_nf = 2
                use_amp = True
            if dump_path:
                if scope in ("all", "model", "full"):
                    self.activation_store.dump_all_possible_hook_points(
                        dump_path,
                        include_prefixes=include_prefixes,
                        exclude_prefixes=exclude_prefixes,
                        max_modules=max_modules,
                        max_items=max_items,
                        with_sizes=with_sizes,
                        tmp_bs=tmp_bs,
                        tmp_nf=tmp_nf,
                        use_amp=use_amp,
                    )
                else:
                    self.activation_store.dump_possible_hook_points(
                        dump_path,
                        with_sizes=with_sizes,
                        max_items=max_items,
                    )

        if enabled and self.activation_store is not None:
            self.activation_store.validate_hook_points(
                strict=strict,
                raise_on_invalid=True,
                log_possible=log_possible,
                max_list=max_list,
            )

    # ---------------------------- SAE create --------------------------------

    def _create_sae_models(self):
        sae_cfg_all = self.config["sae"]
        layers_to_train = self.activation_store.expanded_hook_points
        if self.rank == 0:
            logger.info(f"Total hook targets discovered: {layers_to_train}")

        # 레이어 오너 매핑(전 rank 동일하게 결정)
        owners = self.layer_owners
        if not owners:
            logger.warning("Owner map missing in pipeline; assigning now as fallback.")
            self._assign_layer_owners()
            owners = self.layer_owners
            

        # 이 랭크가 담당할 레이어(오너=내 rank)
        # 이 랭크가 담당할 레이어(오너=내 rank)
        self.my_layers_to_train = [ln for ln in layers_to_train if owners.get(ln, -1) == self.rank]
        logger.info(f"[Rank {self.rank}] owner of {len(self.my_layers_to_train)} layers: {self.my_layers_to_train}")

        # === W&B metric 정의(레이어 확정 후) ===
        if self.wandb_run:
            for lname in self.my_layers_to_train:
                safe = lname.replace("/", "_")
                # step 축
                self.wandb_run.define_metric(f"{safe}/step")
                # token 축
                self.wandb_run.define_metric(f"{safe}/token_seen")
                
                # 접두 네임스페이스로 축 매핑
                self.wandb_run.define_metric(f"{safe}/by_step/*",  step_metric=f"{safe}/step",       step_sync=True)
                self.wandb_run.define_metric(f"{safe}/by_token/*", step_metric=f"{safe}/token_seen", step_sync=True)

                # 요약 통계(스텝 축 지표 기준)
                self.wandb_run.define_metric(f"{safe}/by_step/l2",   summary="min")
                self.wandb_run.define_metric(f"{safe}/by_step/loss", summary="min")
                # validation 지표(step 축 공유)
                self.wandb_run.define_metric(f"{safe}/val/*", step_metric=f"{safe}/step", step_sync=True)

                # 초기값 0 (축 고정)
                self._wandb_log({f"{safe}/step": 0, f"{safe}/token_seen": 0})

        available_types = list_available_saes()
        if self.rank == 0:
            logger.info(f"Available SAE types: {available_types}")
        device = self.device

        # SAE 인스턴스 생성(오너 레이어만)
        for layer_name in self.my_layers_to_train:
            act_size = self.activation_store.get_activation_size(layer_name)
            tr = sae_cfg_all["training"]
            layer_over = tr.get("per_layer", {}).get(layer_name, {})
            sae_type = layer_over.get("sae_type", tr.get("sae_type", "batch-topk")).lower()

            expansion = layer_over.get("expansion_factor", tr.get("expansion_factor", 8))
            sae_cfg = {
                "act_size": act_size,
                "dict_size": layer_over.get("dict_size", act_size * expansion),
                "device": str(device),
                "dtype": tr.get("dtype", "float32"),
                "seed": tr.get("seed", 42),
                "l1_coeff": layer_over.get("l1_coeff", tr.get("l1_coeff", 0.0)),
                "input_unit_norm": tr.get("input_unit_norm", True),
                "n_batches_to_dead": tr.get("n_batches_to_dead", 20),
                # Common hyperparams (global defaults, overridable by per-layer)
                "k": tr.get("k", 32),
                "k_aux": tr.get("k_aux", 512),
                "aux_frac": tr.get("aux_frac", 0.1),
                # Loss selection & penalty (read by _resolve_loss_name / _get_aux_penalty)
                "loss_name": layer_over.get("loss_name", tr.get("loss_name", "mse_l1")),
                "aux_penalty": layer_over.get("aux_penalty", tr.get("aux_penalty", 0.1)),
                # Freq / bias monitoring & penalty
                "freq_ema_decay": tr.get("freq_ema_decay", 0.999),
                "spatial_var_penalty": tr.get("spatial_var_penalty", 0.0),
                "spatial_var_freq_threshold": tr.get("spatial_var_freq_threshold", 0.01),
                "freq_penalty_coeff": tr.get("freq_penalty_coeff", 0.0),
                "bev_penalty": tr.get("bev_penalty", 0.0),
                # RA-SAE specific
                "input_global_center_norm": tr.get("input_global_center_norm", False),
                "delta": tr.get("delta", 0.5),
                "reanim_coeff": tr.get("reanim_coeff", 0.0),
                "nmse_weight": tr.get("nmse_weight", 0.0),
                **{k: v for k, v in layer_over.items() if k not in ("sae_type",)},
            }

            # RA-SAE specific: load cluster centers
            _ra_types = [
                "ra-topk", "ra-ar", "ra-jump", "ra-jumprelu",
                "ra-batchtopk", "ra-unitcentroid-batchtopk",
            ]
            if sae_type in _ra_types:
                points = self._load_kmeans_centers(layer_name, act_size)
                sae_cfg["points"] = points
                # Load global_mean for unit-centroid variant
                if sae_type == "ra-unitcentroid-batchtopk":
                    global_mean = self._load_global_mean(layer_name, act_size)
                    if global_mean is not None:
                        sae_cfg["global_mean"] = global_mean

            if sae_type in ["topk", "top_k"]:
                sae_cfg.update({"top_k": tr["k"], "top_k_aux": tr.get("k_aux", 512), "aux_frac": tr.get("aux_frac", 1/32)})
                sae = create_sae("topk", sae_cfg)
            elif sae_type in ["vanilla", "standard"]:
                sae = create_sae("vanilla", sae_cfg)
            elif sae_type in ["batch-topk", "batchtopk", "batch_topk"]:
                sae_cfg.update({"k": tr["k"], "k_aux": tr.get("k_aux", 512), "aux_frac": tr.get("aux_frac", 1/32),
                                "batch_size": tr.get("activation_batch_size", 4096)})
                sae = create_sae("batch-topk", sae_cfg)
            elif sae_type in ["matryoshka"]:
                sae_cfg.update({"group_sizes": tr.get("group_sizes", [sae_cfg["dict_size"]]),
                                "k": tr["k"], "aux_frac": tr.get("aux_frac", 1/32)})
                sae = create_sae("matryoshka", sae_cfg)
            else:
                sae = create_sae(sae_type, sae_cfg)

            sae.to(device)
            sae.train()

            # b_dec initialization from global_mean
            b_dec_init = layer_over.get("b_dec_init", tr.get("b_dec_init", "zeros"))
            if b_dec_init == "mean" and sae_type in _ra_types:
                global_mean = self._load_global_mean(layer_name, act_size)
                if global_mean is not None:
                    sae.b_dec.data.copy_(global_mean.to(device))
                    logger.info(f"[Rank {self.rank}] Initialized b_dec from global_mean for {layer_name}")
                else:
                    logger.warning(
                        f"[Rank {self.rank}] b_dec_init='mean' but no global_mean found for {layer_name}. "
                        "Using zeros. Run K-means with --unit-norm first."
                    )

            # b_norm initialization from centroid mean (mean direction on unit sphere)
            if hasattr(sae, "b_norm") and sae_type in _ra_types:
                points = sae_cfg.get("points")
                if points is not None:
                    centroid_mean = points.mean(dim=0)
                    sae.b_norm.data.copy_(centroid_mean.to(device))
                    logger.info(f"[Rank {self.rank}] Initialized b_norm from centroid mean for {layer_name}")

            self.sae_models[layer_name] = sae
            logger.info(f"[Rank {self.rank}] Created SAE({sae_type}) for {layer_name} (act={act_size})")

    def _load_kmeans_centers(self, layer_name: str, act_size: int) -> torch.Tensor:
        """Load pre-computed K-means cluster centers for RA-SAE initialization.

        Expected path: ``{centroids_dir}/{sanitize_layer_name(layer_name)}/centroids.pt``

        Uses shared utilities from :mod:`src.core.sae.kmeans.utils`.
        """
        tr = self.config["sae"]["training"]
        kmeans_cfg = tr.get("kmeans_init", {})

        if not kmeans_cfg.get("enabled", False):
            raise ValueError(
                f"RA-SAE ({layer_name}) requires K-means initialization. "
                "Set sae.training.kmeans_init.enabled=true in config."
            )

        base_dir = Path(kmeans_cfg.get("centroids_dir", "outputs/kmeans_centers"))
        path = centroids_path(base_dir, layer_name)

        if not path.exists():
            raise FileNotFoundError(
                f"K-means centroids not found: {path}\n"
                f"Run 'python scripts/run_kmeans.py --config <config>' first."
            )

        centroids = load_centroids(path)

        # Validate dimensions
        if centroids.shape[1] != act_size:
            raise ValueError(
                f"Centroid dimension mismatch for {layer_name}: "
                f"expected {act_size}, got {centroids.shape[1]}"
            )

        logger.info(
            f"[Rank {self.rank}] Loaded {centroids.shape[0]} K-means centers "
            f"for {layer_name}"
        )

        return centroids

    def _load_global_mean(self, layer_name: str, act_size: int) -> Optional[torch.Tensor]:
        """Load global_mean from centroid checkpoint (saved by K-means --unit-norm)."""
        tr = self.config["sae"]["training"]
        kmeans_cfg = tr.get("kmeans_init", {})
        base_dir = Path(kmeans_cfg.get("centroids_dir", "outputs/kmeans_centers"))
        path = centroids_path(base_dir, layer_name)
        if not path.exists():
            return None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        gm = ckpt.get("global_mean", None)
        if gm is not None:
            if gm.shape[0] != act_size:
                raise ValueError(
                    f"global_mean dimension mismatch for {layer_name}: "
                    f"expected {act_size}, got {gm.shape[0]}"
                )
            logger.info(f"[Rank {self.rank}] Loaded global_mean for {layer_name}")
        else:
            logger.warning(
                f"[Rank {self.rank}] No global_mean in centroid checkpoint for {layer_name}. "
                "Was K-means trained with --unit-norm?"
            )
        return gm

    # ------------------------------- validation ---------------------------------
    def _layers_for_rank(self) -> list[str]:
        if self.layer_owners:
            return [ln for ln, owner in self.layer_owners.items() if owner == self.rank]
        return list(getattr(self.activation_store, "expanded_hook_points", []))

    def _reset_store_buffers(self, store: UniversalActivationStore, *, reset_counters: bool = True) -> None:
        for k in list(store.activations.keys()):
            store.activations[k] = []
        if hasattr(store, "_prov_accum"):
            for k in list(store._prov_accum.keys()):
                store._prov_accum[k] = []
        for q in store.queues.values():
            q.blocks.clear()
            q.ntoks = 0
        if getattr(store, "prov_queues", None):
            for pq in store.prov_queues.values():
                pq.blocks.clear()
                pq.ntoks = 0
        for mb in getattr(store, "mix_buffers", {}).values():
            mb.clear()
        if reset_counters:
            if hasattr(store, "queue_init_tokens"):
                store.queue_init_tokens.clear()
            if hasattr(store, "queue_peak_tokens"):
                store.queue_peak_tokens.clear()

    def _build_validation_loader(self) -> DataLoader | None:
        if not self.validation_cfg.get("enabled", False):
            return None
        dcfg = dict(self.validation_cfg.get("dataset") or {})
        if not dcfg:
            dcfg = dict(self.config.get("dataset", {}))
        if "split" not in dcfg and "split" in self.validation_cfg:
            dcfg["split"] = self.validation_cfg["split"]
        if "root" not in dcfg and "root" in self.validation_cfg:
            dcfg["root"] = self.validation_cfg["root"]
        dcfg.setdefault("is_train", False)
        dcfg.setdefault("shuffle", False)

        builder_path = dcfg.get("builder") or self.config.get("dataset", {}).get("builder")
        if not builder_path:
            logger.warning("Validation is enabled but no dataset builder was provided.")
            return None

        builder = load_obj(builder_path)
        per_rank = bool(self.validation_cfg.get("per_rank", True))
        v_rank = 0 if per_rank else self.rank
        v_world = 1 if per_rank else self.world_size
        result = builder(
            dataset_cfg=dcfg,
            rank=v_rank,
            world_size=v_world,
            device=self.device,
            full_config=self.config,
        )

        dataset = collate_fn = sampler = None
        if isinstance(result, dict):
            dataset = result.get("dataset")
            collate_fn = result.get("collate_fn")
            sampler = result.get("sampler")
        elif isinstance(result, (list, tuple)):
            if len(result) >= 2:
                dataset, collate_fn = result[:2]
            if len(result) >= 3:
                sampler = result[2]
        else:
            dataset = result

        if dataset is None or collate_fn is None:
            logger.warning("Validation dataset builder must return both dataset and collate_fn.")
            return None

        pin = self.device.type == "cuda" and self.config.get("sae", {}).get("training", {}).get("pin_memory", True)
        bs = int(dcfg.get("batch_size", self.config.get("dataset", {}).get("batch_size", 1)))
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=bool(dcfg.get("shuffle", False)) if sampler is None else False,
            sampler=sampler,
            num_workers=int(dcfg.get("num_workers", 0)),
            pin_memory=pin,
            drop_last=bool(dcfg.get("drop_last", False)),
            collate_fn=collate_fn,
        )
        return loader

    def _drain_validation_cache(self, *, max_tokens: int, batch_tokens: int) -> dict[str, torch.Tensor]:
        caches: dict[str, torch.Tensor] = {}
        target_layers = self._layers_for_rank()
        for lname in target_layers:
            total = 0
            pieces: list[torch.Tensor] = []
            while True:
                want = batch_tokens if batch_tokens > 0 else None
                acts = self.activation_store.next_batch(lname, batch_size=want)
                if acts is None or acts.numel() == 0:
                    break
                chunk = acts.detach().to("cpu")
                if max_tokens > 0 and total + chunk.shape[0] > max_tokens:
                    chunk = chunk[: max_tokens - total]
                pieces.append(chunk)
                total += chunk.shape[0]
                if max_tokens > 0 and total >= max_tokens:
                    break
            if pieces:
                caches[lname] = torch.cat(pieces, dim=0).contiguous()
        return caches

    def _prepare_validation_cache(self) -> None:
        if not self.validation_cfg.get("enabled", False):
            return
        if self.activation_store is None:
            return

        loader = self._build_validation_loader()
        if loader is None:
            logger.warning("[Rank %d] Validation loader missing; skip validation caching.", self.rank)
            return

        max_batches = int(self.validation_cfg.get("max_batches", 0))
        max_tokens = int(self.validation_cfg.get("max_tokens_per_layer", 131072))
        cache_batch = int(
            self.validation_cfg.get(
                "cache_batch_tokens",
                getattr(self.activation_store, "activation_batch_size", 4096),
            )
        )

        self._reset_store_buffers(self.activation_store, reset_counters=False)
        batches = 0
        with torch.no_grad():
            for batch in loader:
                if max_batches and batches >= max_batches:
                    break
                self.activation_store._run_model_on_batch(batch)
                batches += 1
        self.activation_store._flush_activations_to_queues()

        self.validation_cache = self._drain_validation_cache(
            max_tokens=max_tokens,
            batch_tokens=cache_batch,
        )
        self.validation_ready = bool(self.validation_cache)

        self._reset_store_buffers(self.activation_store, reset_counters=True)
        if dist.is_initialized():
            self._barrier()

        if self.rank == 0:
            logger.info(
                "Validation cache %s (layers=%d, batches=%d, max_tokens=%s)",
                "ready" if self.validation_ready else "empty",
                len(self.validation_cache),
                batches,
                "unlimited" if max_tokens <= 0 else max_tokens,
            )

    def _extract_losses(self, out) -> tuple[float, float, float, float]:
        if isinstance(out, dict):
            loss = float(out.get("loss", torch.tensor(0.0)).item())
            l1 = float(out.get("l1_loss", torch.tensor(0.0)).item())
            l2 = float(out.get("l2_loss", torch.tensor(0.0)).item())
            aux = float(out.get("aux_loss", torch.tensor(0.0)).item())
            return loss, l1, l2, aux
        if torch.is_tensor(out):
            val = float(out.item())
        else:
            val = float(out)
        return val, 0.0, val, 0.0

    def _run_validation(self, steps_done: Dict[str, int]) -> None:
        if not (self.validation_ready and self.validation_cfg.get("enabled", False)):
            return
        if not self.wandb_run:
            return
        eval_bs = int(
            self.validation_cfg.get(
                "eval_batch_tokens",
                getattr(self.activation_store, "activation_batch_size", 4096),
            )
        )
        global_step = max(steps_done.values()) if steps_done else 0
        logged_any = False

        for lname, cache in self.validation_cache.items():
            sae = self.sae_models.get(lname)
            if sae is None:
                continue
            was_train = sae.training
            sae.eval()

            total = 0
            sum_loss = sum_l1 = sum_l2 = sum_aux = 0.0
            max_l2 = 0.0
            with torch.no_grad():
                for start in range(0, cache.shape[0], eval_bs):
                    chunk = cache[start : start + eval_bs].to(self.device, non_blocking=True)
                    out = sae(chunk)
                    loss, l1, l2, aux = self._extract_losses(out)
                    n = chunk.shape[0]
                    total += n
                    sum_loss += loss * n
                    sum_l1 += l1 * n
                    sum_l2 += l2 * n
                    sum_aux += aux * n
                    max_l2 = max(max_l2, l2)
            if was_train:
                sae.train()
            if total == 0:
                continue

            safe = lname.replace("/", "_")
            payload = {
                "global/step": global_step,
                f"{safe}/step": steps_done.get(lname, 0),
                f"{safe}/val/loss": sum_loss / total,
                f"{safe}/val/l2_mean": sum_l2 / total,
                f"{safe}/val/l2_max": max_l2,
                f"{safe}/val/l1_mean": sum_l1 / total,
                f"{safe}/val/aux_mean": sum_aux / total,
                f"{safe}/val/tokens": total,
            }
            self._wandb_log(payload, commit=False)
            logged_any = True

        if logged_any:
            self._wandb_log({}, commit=True)

    # ------------------------------- ckpt --------------------------------
    def _save_layer_ckpt(self, lname: str, step: int, opt, tokens_seen: int):
        save_root = Path(self.config["sae"]["output"]["save_path"])
        layer_dir = save_root / sanitize_layer_name(lname)
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / f"step_{step:07d}_tokens_{tokens_seen}.pt"

        sae_config = dict(self.sae_models[lname].config)
        # Ensure sae_type is saved for correct reload
        if "sae_type" not in sae_config or not sae_config["sae_type"]:
            tr_cfg = self.config.get("sae", {}).get("training", {})
            sae_config["sae_type"] = tr_cfg.get("sae_type", "batch-topk")
        pkg = {
            "layer_name": lname,
            "owner_rank": self.rank,
            "world_size": self.world_size,
            "step": step,
            "tokens_seen": tokens_seen,
            "sae_state": self.sae_models[lname].state_dict(),
            "optimizer_state": opt.state_dict(),
            "sae_config": sae_config,
            "act_size": self.activation_store.get_activation_size(lname),
            "saved_at": time.time(),
        }
        torch.save(pkg, path)
        logger.info(f"[Rank {self.rank}] Saved ckpt: {path}")

        # Delete old checkpoints, keeping only the most recent N
        keep = int(self.config.get("sae", {}).get("training", {}).get("keep_last_ckpts", 0))
        if keep > 0:
            ckpts = sorted(layer_dir.glob("step_*.pt"))
            if len(ckpts) > keep:
                for old in ckpts[:-keep]:
                    old.unlink()
                    logger.info(f"[Rank {self.rank}] Deleted old ckpt: {old.name}")

        # 선택: W&B artifact 업로드
        if self.wandb_run and self.wandb_cfg.get("upload_checkpoints", False):
            art = wandb.Artifact(name=f"sae_{lname.replace(':','__').replace('/','_')}", type="model")
            art.add_file(str(path))
            self.wandb_run.log_artifact(art)


    # ----------------------------- warmup from cache --------------------------------

    def _run_warmup_from_cache(self, opt_for, sched_for, steps_done):
        """Train SAEs from pre-extracted K-means activation cache files before
        the main streaming loop.

        Reads ``sae.training.warmup`` config section.  For each owned layer,
        iterates over ``chunk_*.pt`` files in the cache directory, shuffles
        them (and tokens within), and performs standard SAE training steps.

        The optimizer, scheduler, and ``steps_done`` / ``tokens_seen`` counters
        are shared with the main loop so warmup progress carries over.
        """
        tr_cfg = self.config["sae"]["training"]
        warmup_cfg = tr_cfg.get("warmup") or {}

        if not warmup_cfg.get("enabled", False):
            return

        data_dir = Path(warmup_cfg["data_dir"])
        if not data_dir.exists():
            logger.warning(
                "[Rank %d] Warmup data_dir does not exist: %s — skipping warmup.",
                self.rank, data_dir,
            )
            return

        epochs = int(warmup_cfg.get("epochs", 1))
        default_batch_size = int(warmup_cfg.get("batch_size", 4096))
        do_shuffle = bool(warmup_cfg.get("shuffle", True))
        log_every = int(self.wandb_cfg.get("log_every_steps", 10)) if self.wandb_run else 0
        speed_log_sec = float(tr_cfg.get("speed_log_every_sec", 5.0))
        ckpt_every = int(tr_cfg.get("ckpt_every_steps", 0))
        steps_goal = int(tr_cfg.get("num_training_steps", 10000))
        warmup_max_steps = int(warmup_cfg.get("max_steps", steps_goal))

        import random

        logger.info(
            "[Rank %d] Starting warmup from cache: data_dir=%s, epochs=%d, "
            "default_batch_size=%d, shuffle=%s",
            self.rank, data_dir, epochs, default_batch_size, do_shuffle,
        )

        for lname in self.my_layers_to_train:
            # Skip if already past warmup (resumed from checkpoint)
            if steps_done.get(lname, 0) > 0:
                logger.info(
                    "[Rank %d] Skipping warmup for %s — already at step %d",
                    self.rank, lname, steps_done[lname],
                )
                continue

            sae = self.sae_models[lname]
            opt = opt_for[lname]
            sched = sched_for[lname]

            # Determine per-layer batch size
            per_layer_cfg = tr_cfg.get("per_layer", {}).get(lname, {})
            train_bs = int(per_layer_cfg.get("batch_size", default_batch_size))

            # Grad clipping config
            clip_grad = float(per_layer_cfg.get("clip_grad", tr_cfg.get("clip_grad", 1.0)))

            # Locate chunk files for this layer
            safe_lname = sanitize_layer_name(lname)
            layer_dir = data_dir / safe_lname
            if not layer_dir.exists():
                logger.warning(
                    "[Rank %d] Warmup: no directory for %s at %s — skipping.",
                    self.rank, lname, layer_dir,
                )
                continue

            chunk_paths = sorted(layer_dir.glob("chunk_*.pt"))
            if not chunk_paths:
                logger.warning(
                    "[Rank %d] Warmup: no chunk_*.pt files found in %s — skipping.",
                    self.rank, lname,
                )
                continue

            # Estimate total size to decide loading strategy
            max_buffer_mb = float(warmup_cfg.get("max_buffer_mb", 2048))
            buffer_chunks = int(warmup_cfg.get("buffer_chunks", 16))

            # Peek at first chunk to estimate per-chunk size
            first_chunk = torch.load(chunk_paths[0], map_location="cpu", weights_only=True)
            per_chunk_mb = first_chunk.numel() * first_chunk.element_size() / 1e6
            total_mb = per_chunk_mb * len(chunk_paths)
            del first_chunk

            if total_mb <= max_buffer_mb:
                load_mode = "full"
                effective_buf = len(chunk_paths)
            else:
                load_mode = "buffer"
                effective_buf = min(buffer_chunks, len(chunk_paths))

            logger.info(
                "[Rank %d] Warmup %s: %d chunks (%.0f MB total), train_bs=%d, "
                "mode=%s (buf=%d chunks)",
                self.rank, lname, len(chunk_paths), total_mb, train_bs,
                load_mode, effective_buf,
            )

            # Speed tracking
            speed_tokens = 0
            speed_elapsed = 0.0
            speed_t0 = time.time()
            total_warmup_tokens = 0

            for epoch in range(epochs):
                epoch_paths = list(chunk_paths)
                if do_shuffle:
                    random.shuffle(epoch_paths)

                # Process chunks in groups of effective_buf
                for group_start in range(0, len(epoch_paths), effective_buf):
                    if steps_done[lname] >= warmup_max_steps:
                        break

                    group = epoch_paths[group_start:group_start + effective_buf]

                    # Load all chunks in this group into a single tensor
                    parts = []
                    for cpath in group:
                        try:
                            t = torch.load(cpath, map_location="cpu", weights_only=True)
                        except Exception as e:
                            logger.warning(
                                "[Rank %d] Warmup: failed to load %s: %s — skipping.",
                                self.rank, cpath, e,
                            )
                            continue
                        if t.ndim == 2:
                            parts.append(t)
                    if not parts:
                        continue

                    pool = torch.cat(parts, dim=0)
                    del parts

                    # Shuffle the entire pool (cross-chunk mixing)
                    if do_shuffle:
                        pool = pool[torch.randperm(pool.shape[0])]

                    n_pool = pool.shape[0]

                    # Train on batches from the shuffled pool
                    for start in range(0, n_pool, train_bs):
                        if steps_done[lname] >= warmup_max_steps:
                            break

                        end = min(start + train_bs, n_pool)
                        batch = pool[start:end]
                        actual_bs = batch.shape[0]

                        step_t0 = time.time()

                        # Forward / backward (same pattern as main loop)
                        opt.zero_grad(set_to_none=True)
                        acts = batch.to(self.device)
                        out = sae(acts)

                        loss_tensor = out["loss"] if isinstance(out, dict) else out
                        (loss_tensor * (actual_bs / float(train_bs))).backward()

                        if clip_grad > 0:
                            torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad)

                        opt.step()
                        sched.step()
                        if hasattr(sae, "make_decoder_weights_and_grad_unit_norm"):
                            sae.make_decoder_weights_and_grad_unit_norm()

                        steps_done[lname] += 1
                        self.tokens_seen[lname] += actual_bs
                        total_warmup_tokens += actual_bs

                        loss_val, l1_val, l2_val, aux_val = self._extract_losses(out)

                        # Extract quality metrics
                        _rel_l2 = float(out.get("relative_l2", 0)) if isinstance(out, dict) else 0.0
                        _fa = out.get("feature_acts") if isinstance(out, dict) else None
                        _sparsity = float((_fa > 0).float().mean()) if _fa is not None else 0.0

                        # Speed logging
                        step_dt = time.time() - step_t0
                        speed_tokens += actual_bs
                        speed_elapsed += step_dt

                        if speed_log_sec > 0 and (time.time() - speed_t0) >= speed_log_sec:
                            tps = speed_tokens / max(speed_elapsed, 1e-6)
                            print(
                                f"[warmup] {lname} rank={self.rank} "
                                f"epoch={epoch+1}/{epochs} "
                                f"step={steps_done[lname]} "
                                f"tokens/s={tps:,.0f} "
                                f"loss={loss_val:.5f} l2={l2_val:.5f} aux={aux_val:.5f} "
                                f"rel_l2={_rel_l2:.5f} sparsity={_sparsity:.6f}",
                                flush=True,
                            )
                            speed_tokens = 0
                            speed_elapsed = 0.0
                            speed_t0 = time.time()

                        # W&B logging
                        if self.wandb_run and log_every > 0 and (steps_done[lname] % log_every == 0):
                            safe = lname.replace("/", "_")
                            # dead feature stats
                            _cnt = sae.num_batches_not_active
                            _n_dead_th = sae.config.get("n_batches_to_dead", 20)
                            _dead_frac = float((_cnt > _n_dead_th).float().mean().item())
                            _never_frac = float((_cnt >= steps_done[lname]).float().mean().item())

                            payload = {
                                f"{safe}/step": steps_done[lname],
                                f"{safe}/token_seen": self.tokens_seen[lname],
                                f"warmup/{safe}/loss": loss_val,
                                f"warmup/{safe}/l2": l2_val,
                                f"warmup/{safe}/l1": l1_val,
                                f"warmup/{safe}/aux": aux_val,
                                f"warmup/{safe}/epoch": epoch + 1,
                                f"warmup/{safe}/lr": opt.param_groups[0]["lr"],
                                f"warmup/{safe}/dead_frac": _dead_frac,
                                f"warmup/{safe}/never_activated_frac": _never_frac,
                                f"warmup/{safe}/sparsity": _sparsity,
                                f"warmup/{safe}/relative_l2": _rel_l2,
                                f"{safe}/by_step/loss": loss_val,
                                f"{safe}/by_step/l2": l2_val,
                                f"{safe}/by_step/l1": l1_val,
                                f"{safe}/by_step/aux": aux_val,
                                f"{safe}/by_step/lr": opt.param_groups[0]["lr"],
                                f"{safe}/by_step/dead_frac": _dead_frac,
                                f"{safe}/by_step/sparsity": _sparsity,
                                f"{safe}/by_step/relative_l2": _rel_l2,
                                f"{safe}/by_token/loss": loss_val,
                                f"{safe}/by_token/l2": l2_val,
                                f"{safe}/by_token/l1": l1_val,
                                f"{safe}/by_token/aux": aux_val,
                                f"{safe}/by_token/sparsity": _sparsity,
                                f"{safe}/by_token/relative_l2": _rel_l2,
                            }
                            # k_eff for warmup
                            if _fa is not None:
                                _k_eff = (_fa > 0).float().sum(dim=-1).mean().item()
                                payload[f"warmup/{safe}/k_eff"] = _k_eff
                                payload[f"{safe}/by_step/k_eff"] = _k_eff
                                payload[f"{safe}/by_token/k_eff"] = _k_eff
                            # Feature frequency / bias monitoring
                            if hasattr(sae, "feature_freq_ema"):
                                freq = sae.feature_freq_ema.float()
                                alive = freq > 0
                                payload[f"{safe}/freq/max"] = freq.max().item()
                                payload[f"{safe}/freq/mean"] = freq[alive].mean().item() if alive.any() else 0.0
                                payload[f"{safe}/freq/high_frac"] = (freq > 0.1).float().mean().item()
                                payload[f"{safe}/freq/low_frac"] = ((freq > 0) & (freq < 0.01)).float().mean().item()
                                if alive.sum() > 1:
                                    sf = freq[alive].sort().values
                                    n_alive = sf.shape[0]
                                    idx = torch.arange(1, n_alive + 1, dtype=torch.float32, device=sf.device)
                                    gini = (2.0 * (idx * sf).sum() / (n_alive * sf.sum()) - (n_alive + 1.0) / n_alive).item()
                                    payload[f"{safe}/freq/gini"] = max(0.0, gini)
                            self._wandb_log(payload)

                        # Checkpoint saving
                        if ckpt_every > 0 and (steps_done[lname] % ckpt_every == 0):
                            self._save_layer_ckpt(
                                lname, steps_done[lname], opt, self.tokens_seen[lname],
                            )

                    del pool
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if steps_done[lname] >= warmup_max_steps:
                    break

            # --- warmup 완료 후 dead feature 통계 ---
            sae = self.sae_models[lname]
            n_dead_thresh = sae.config.get("n_batches_to_dead", 20)
            cnt = sae.num_batches_not_active
            never_active = int((cnt >= steps_done[lname]).sum().item())  # warmup 내내 한번도 안 쓰인 feature
            dead_now = int((cnt > n_dead_thresh).sum().item())
            total_features = cnt.numel()
            logger.info(
                "[Rank %d] Warmup done for %s: steps=%d, tokens=%s, mode=%s | "
                "features: total=%d, never_activated=%d (%.1f%%), "
                "dead(>%d)=%d (%.1f%%)",
                self.rank, lname, steps_done[lname],
                self._fmt_num(total_warmup_tokens), load_mode,
                total_features,
                never_active, 100.0 * never_active / max(total_features, 1),
                n_dead_thresh,
                dead_now, 100.0 * dead_now / max(total_features, 1),
            )

        # Synchronise ranks before the main loop.
        # Cannot use NCCL barrier here because warmup duration varies
        # greatly between ranks (different layers have different amounts
        # of cached data).  Use the TCPStore instead — no NCCL timeout.
        if dist.is_initialized() and self.world_size > 1:
            store = dist.distributed_c10d._get_default_store()
            store.set(f"warmup_done_{self.rank}", "1")
            logger.info(
                "[Rank %d] Warmup finished locally. Waiting for all %d ranks …",
                self.rank, self.world_size,
            )
            for r in range(self.world_size):
                if r == self.rank:
                    continue
                while True:
                    try:
                        store.get(f"warmup_done_{r}")
                        break
                    except Exception:
                        time.sleep(5.0)

        logger.info("[Rank %d] Warmup phase complete.", self.rank)

    # ----------------------------- train loop --------------------------------

    def train(self):
        # 1) 모델/데이터/스토어
        self.model = self._load_model()
        self._load_dataset()
        self._create_activation_store()

        # 1) discover로 expanded_hook_points 확정
        self.activation_store.discover_hook_points()
        self._log_gpu_memory("after_discover")

        # 2) 오너 매핑을 모든 rank에서 동일하게 계산/주입
        self._assign_layer_owners()
        
        # 2) W&B 설정(필요시)
        self._setup_wandb()

        # 2.5) validation 캐시 준비(선택)
        self._prepare_validation_cache()
        self._log_gpu_memory("after_validation_cache")

        
        # 3) 이제 prefill
        prefill_n = int(self.config["sae"]["training"].get("prefill_batches", 2))
        if prefill_n > 0:
            self.activation_store.collect_round(n_batches=prefill_n)
            self._print_buffer_progress(force=True)
            self._log_gpu_memory("after_prefill")

        # 4) SAE 생성(오너만 해당 레이어 SAE를 만듦)
        self._create_sae_models()
        self._log_gpu_memory("after_create_sae")

        # 5) 옵티마이저
        tr_cfg = self.config["sae"]["training"]
        lr = float(tr_cfg.get("lr", 1e-4))
        
        ckpt_every = int(tr_cfg.get("ckpt_every_steps", 0))  # 0이면 끔
        log_every = int(self.wandb_cfg.get("log_every_steps", 10)) if self.wandb_run else 0
        hist_every = int(self.wandb_cfg.get("log_hist_every_steps", 1000)) if self.wandb_run else 0
        speed_log_sec = float(tr_cfg.get("speed_log_every_sec", 0.0))
        
        opt_for = {lname: torch.optim.Adam(self.sae_models[lname].parameters(), lr=lr)
                   for lname in self.my_layers_to_train}

        steps_goal = int(tr_cfg.get("num_training_steps", 10000))  # 레이어별 목표 step

        # 5-b) CosineScheduler (warmup + cosine decay)
        lr_final = float(tr_cfg.get("lr_final", 0.0))
        warmup_steps = int(tr_cfg.get("warmup_steps", 0))
        sched_for = {
            lname: CosineScheduler(
                optimizer=opt_for[lname],
                base_value=lr,
                final_value=lr_final,
                total_iters=steps_goal,
                warmup_iters=warmup_steps,
                start_warmup_value=0.0,
            )
            for lname in self.my_layers_to_train
        }
        steps_done: Dict[str, int] = {ln: 0 for ln in self.my_layers_to_train}
        self.tokens_seen = {ln: 0 for ln in self.my_layers_to_train}
        speed_track = {ln: {"tokens": 0, "elapsed": 0.0, "steps": 0} for ln in self.my_layers_to_train}
        self._resume_from_checkpoints(opt_for, steps_done)
        # Fast-forward schedulers to resumed step
        for lname in self.my_layers_to_train:
            sched_for[lname].iter = steps_done[lname]
        # === Offline warmup from pre-extracted cache ===
        self._log_gpu_memory("before_warmup")
        self._run_warmup_from_cache(opt_for, sched_for, steps_done)
        self._log_gpu_memory("after_warmup")

        val_every = int(self.validation_cfg.get("every_steps", 0))
        last_val_logged = -val_every if val_every > 0 else 0
        max_steps_per_cycle = int(tr_cfg.get("max_steps_per_cycle", 1))

        # 학습 루프
        while True:
            any_trained_local = False
            global_step = max(steps_done.values()) if steps_done else 0

            # 오너 레이어에 대해서만 학습 시도
            for lname in list(self.my_layers_to_train):
                if steps_done[lname] >= steps_goal:
                    continue

                sae = self.sae_models[lname]
                opt = opt_for[lname]

                cfg_l = self.activation_store._cfg_for_layer(lname)
                train_bs = int(cfg_l.get("batch_size", self.activation_store.activation_batch_size))  # SAE 목표 배치(토큰) 수
                micro    = int(cfg_l.get("micro", train_bs))  # micro가 없으면 train_bs와 동일 → 기존 동작
                # micro로 몇 번 쪼개서 누적할지(기본: ceil(train_bs / micro)), 사용자가 'accum'을 주면 우선
                default_accum = (train_bs + micro - 1) // micro
                accum    = int(cfg_l.get("accum", default_accum))

                # 큐에 남은 토큰 기반으로 이번 사이클에 최대 몇 스텝을 돌릴지 결정
                q = getattr(self.activation_store, "queues", {}).get(lname) if self.activation_store else None
                queue_tokens = int(getattr(q, "ntoks", 0))
                # Mixing buffer에 있는 토큰도 available로 카운트
                mb = getattr(self.activation_store, "mix_buffers", {}).get(lname)
                if mb is not None:
                    queue_tokens += mb.n_buffered + mb.n_ready * train_bs
                steps_budget = max_steps_per_cycle
                if steps_budget <= 0:
                    steps_budget = queue_tokens // train_bs if train_bs > 0 else 0
                elif queue_tokens > 0:
                    steps_budget = min(steps_budget, max(1, queue_tokens // max(train_bs, 1)))
                steps_budget = max(1, steps_budget)

                steps_this_cycle = 0
                while steps_done[lname] < steps_goal and steps_this_cycle < steps_budget:
                    step_t0 = time.time()
                    opt.zero_grad(set_to_none=True)
                    tokens_accum = 0
                    chunks_done  = 0
                    
                    # 집계용(마이크로 합산)
                    sum_loss = 0.0; sum_l1 = 0.0; sum_l2 = 0.0; sum_aux = 0.0; sum_freq = 0.0
                    sum_svar = 0.0; sum_bev_loss = 0.0
                    sum_n = 0; last_thr = None

                    # sparsity(스텝 평균), k_eff(샘플당 비영 활성 수), 양의 활성 평균 크기
                    nz_total = 0; elts_total = 0
                    k_eff_sum = 0.0
                    pos_sum = 0.0; pos_count = 0

                    # reconstruction quality metrics
                    expvar_sum = 0.0; nmse_sum = 0.0; rel_l2_sum = 0.0

                    # spatial variance diagnostics (last micro-batch snapshot)
                    _last_bev = None; _last_nhf = None

                    while tokens_accum < train_bs and chunks_done < accum:
                        need = min(micro, train_bs - tokens_accum)
                        acts = self.activation_store.next_batch(lname, batch_size=need)
                        if acts is None or acts.numel() == 0:
                            break
                        out = sae(acts)

                        # sparsity / k_eff / pos_mean 집계
                        if isinstance(out, dict) and "feature_acts" in out:
                            fa = out["feature_acts"]
                            nz = (fa > 0)
                            nz_total   += nz.sum().item()
                            elts_total += fa.numel()

                            # k_eff: 샘플당 비영 활성 수(배치 평균)
                            k_eff_batch = nz.sum(dim=-1).float().mean().item()
                            k_eff_sum += k_eff_batch * acts.shape[0]  # n으로 가중 평균

                            # 양의 활성 평균 크기
                            if nz.any():
                                pos_sum   += fa[nz].float().sum().item()
                                pos_count += int(nz.sum().item())

                            thr_val = out.get("threshold", torch.tensor(0.))
                            last_thr = float(thr_val.mean().item()) if thr_val.numel() > 1 else float(thr_val.item())

                            # reconstruction quality metrics
                            if "explained_var" in out and out["explained_var"] is not None:
                                expvar_sum += float(out["explained_var"]) * acts.shape[0]
                            if "nmse" in out and out["nmse"] is not None:
                                nmse_sum += float(out["nmse"]) * acts.shape[0]
                            if "relative_l2" in out and out["relative_l2"] is not None:
                                rel_l2_sum += float(out["relative_l2"]) * acts.shape[0]

                        # 손실 backward (청크 크기 비율만큼 스케일)
                        loss = out["loss"] if isinstance(out, dict) else out
                        (loss * (acts.shape[0] / float(train_bs))).backward()

                        # 집계
                        n = acts.shape[0]
                        sum_n += n
                        tokens_accum += n
                        chunks_done  += 1

                        # 평균용 누적
                        loss_val = float((out["loss"] if isinstance(out, dict) else out).item())
                        sum_loss += loss_val * n
                        if isinstance(out, dict):
                            sum_l1  += float(out.get("l1_loss",  torch.tensor(0.)).item()) * n
                            sum_l2  += float(out.get("l2_loss",  torch.tensor(0.)).item()) * n
                            sum_aux += float(out.get("aux_loss", torch.tensor(0.)).item()) * n
                            sum_freq += float(out.get("freq_loss", torch.tensor(0.)).item()) * n
                            sum_svar += float(out.get("spatial_var_loss", torch.tensor(0.)).item()) * n
                            sum_bev_loss += float(out.get("bev_loss", torch.tensor(0.)).item()) * n

                        # Capture diagnostic metrics before freeing
                        if isinstance(out, dict):
                            _last_bev = float(out.get("bias_explained_var", torch.tensor(0.)).item()) if "bias_explained_var" in out else None
                            _last_nhf = int(out.get("n_high_freq", 0)) if "n_high_freq" in out else None

                        # GPU 메모리 즉시 해제 (연산 그래프 참조 제거)
                        del acts, out, loss

                    if tokens_accum == 0:
                        break

                    clip_grad = float(cfg_l.get("clip_grad", tr_cfg.get("clip_grad", 1.0)))
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad)

                    opt.step()
                    sched_for[lname].step()
                    if hasattr(sae, "make_decoder_weights_and_grad_unit_norm"):
                        sae.make_decoder_weights_and_grad_unit_norm()

                    # === Bias feature folding ===
                    fold_every = int(cfg_l.get("fold_every_steps", tr_cfg.get("fold_every_steps", 0)))
                    if fold_every > 0 and hasattr(sae, "fold_bias_features") and steps_done[lname] % fold_every == 0:
                        fold_freq_thr = float(cfg_l.get("fold_freq_threshold", tr_cfg.get("fold_freq_threshold", 0.95)))
                        fold_cv_thr = float(cfg_l.get("fold_cv_threshold", tr_cfg.get("fold_cv_threshold", 0.02)))
                        n_folded = sae.fold_bias_features(freq_threshold=fold_freq_thr, cv_threshold=fold_cv_thr)
                        if n_folded > 0:
                            logger.info(f"[fold] {lname} step={steps_done[lname]+1}: folded {n_folded} bias features into b_norm")
                            if self.wandb_run:
                                safe_f = lname.replace("/", "_")
                                self._wandb_log({f"{safe_f}/fold/n_folded": n_folded})

                    steps_done[lname] += 1
                    steps_this_cycle += 1
                    self.tokens_seen[lname] += tokens_accum
                    any_trained_local = True

                    # 공통 집계 값
                    denom = max(1, sum_n)
                    mean_loss = sum_loss / denom
                    mean_l2   = sum_l2   / denom
                    mean_l1   = sum_l1   / denom
                    mean_aux  = sum_aux  / denom
                    mean_freq = sum_freq / denom

                    # === 속도 로그(원하면 tqdm 대체) ===
                    if speed_log_sec > 0:
                        st = speed_track.get(lname)
                        if st is None:
                            st = {"tokens": 0, "elapsed": 0.0, "steps": 0}
                            speed_track[lname] = st
                        st["tokens"] += tokens_accum
                        st["steps"]  += 1
                        st["elapsed"] += (time.time() - step_t0)
                        if st["elapsed"] >= speed_log_sec:
                            qtok = int(getattr(q, "ntoks", 0)) if q is not None else 0
                            tps = st["tokens"] / max(st["elapsed"], 1e-6)
                            print(
                                f"[speed] {lname} rank={self.rank} step={steps_done[lname]} "
                                f"tokens/s={tps:,.0f} queue={qtok} loss={mean_loss:.5f}",
                                flush=True,
                            )
                            st["tokens"] = 0
                            st["steps"] = 0
                            st["elapsed"] = 0.0

                    # === wandb 로깅 ===
                    if self.wandb_run and (steps_done[lname] % max(1, log_every) == 0):
                        safe = lname.replace("/", "_")
                        sparsity  = (nz_total / max(1, elts_total))
                        k_eff     = (k_eff_sum / denom)
                        pos_mean  = (pos_sum / max(1, pos_count))
                        # dead 비율: n_batches_to_dead 이상 죽은 피처의 비율
                        try:
                            dead_mask = (sae.num_batches_not_active >= sae.config.get("n_batches_to_dead", 20))
                            dead_frac = float(dead_mask.float().mean().item())
                        except Exception:
                            dead_frac = 0.0

                        payload = {
                            f"{safe}/step": steps_done[lname],
                            f"{safe}/token_seen": self.tokens_seen[lname],

                            # step 축(접두 네임스페이스)
                            f"{safe}/by_step/loss":       mean_loss,
                            f"{safe}/by_step/l2":         mean_l2,
                            f"{safe}/by_step/l1":         mean_l1,
                            f"{safe}/by_step/aux":        mean_aux,
                            f"{safe}/by_step/sparsity":   sparsity,
                            f"{safe}/by_step/k_eff":      k_eff,
                            f"{safe}/by_step/pos_act_mean": pos_mean,
                            f"{safe}/by_step/dead_frac":  dead_frac,
                            f"{safe}/by_step/threshold":  last_thr if last_thr is not None else 0.0,
                            f"{safe}/by_step/lr":         opt.param_groups[0]["lr"],

                            # token 축(접두 네임스페이스)
                            f"{safe}/by_token/loss":       mean_loss,
                            f"{safe}/by_token/l2":         mean_l2,
                            f"{safe}/by_token/l1":         mean_l1,
                            f"{safe}/by_token/aux":        mean_aux,
                            f"{safe}/by_token/sparsity":   sparsity,
                            f"{safe}/by_token/k_eff":      k_eff,
                            f"{safe}/by_token/pos_act_mean": pos_mean,
                            f"{safe}/by_token/dead_frac":  dead_frac,
                            f"{safe}/by_token/threshold":  last_thr if last_thr is not None else 0.0,
                        }  


                        # reconstruction quality metrics
                        if denom > 0:
                            if expvar_sum != 0:
                                mean_expvar = expvar_sum / denom
                                payload[f"{safe}/by_step/explained_var"] = mean_expvar
                                payload[f"{safe}/by_token/explained_var"] = mean_expvar
                            if nmse_sum != 0:
                                mean_nmse = nmse_sum / denom
                                payload[f"{safe}/by_step/nmse"] = mean_nmse
                                payload[f"{safe}/by_token/nmse"] = mean_nmse
                            if rel_l2_sum != 0:
                                mean_rel_l2 = rel_l2_sum / denom
                                payload[f"{safe}/by_step/relative_l2"] = mean_rel_l2
                                payload[f"{safe}/by_token/relative_l2"] = mean_rel_l2

                        # freq penalty loss
                        if mean_freq > 0:
                            payload[f"{safe}/by_step/freq_loss"] = mean_freq
                            payload[f"{safe}/by_token/freq_loss"] = mean_freq

                        # spatial variance penalty
                        mean_svar = sum_svar / denom
                        if mean_svar > 0:
                            payload[f"{safe}/by_step/spatial_var_loss"] = mean_svar
                            payload[f"{safe}/by_token/spatial_var_loss"] = mean_svar

                        # bev penalty loss
                        mean_bev_loss = sum_bev_loss / denom
                        if mean_bev_loss > 0:
                            payload[f"{safe}/by_step/bev_loss"] = mean_bev_loss
                            payload[f"{safe}/by_token/bev_loss"] = mean_bev_loss

                        # bias-only explained variance (from last micro-batch)
                        if _last_bev is not None:
                            payload[f"{safe}/by_step/bias_explained_var"] = _last_bev
                            payload[f"{safe}/by_token/bias_explained_var"] = _last_bev
                        if _last_nhf is not None:
                            payload[f"{safe}/freq/n_high_freq"] = _last_nhf

                        # --- Feature frequency monitoring ---
                        if hasattr(sae, "feature_freq_ema"):
                            freq = sae.feature_freq_ema.float()
                            alive = freq > 0
                            payload[f"{safe}/freq/max"] = freq.max().item()
                            payload[f"{safe}/freq/mean"] = freq[alive].mean().item() if alive.any() else 0.0
                            payload[f"{safe}/freq/high_frac"] = (freq > 0.1).float().mean().item()
                            payload[f"{safe}/freq/low_frac"] = ((freq > 0) & (freq < 0.01)).float().mean().item()
                            # Gini coefficient (inequality of feature usage)
                            if alive.sum() > 1:
                                sf = freq[alive].sort().values
                                n_alive = sf.shape[0]
                                idx = torch.arange(1, n_alive + 1, dtype=torch.float32, device=sf.device)
                                gini = (2.0 * (idx * sf).sum() / (n_alive * sf.sum()) - (n_alive + 1.0) / n_alive).item()
                                payload[f"{safe}/freq/gini"] = max(0.0, gini)

                        self._wandb_log(payload)

                    # === console 로깅 (wandb 없어도) ===
                    _console_every = 100
                    if steps_done[lname] % _console_every == 0:
                        _sparsity = (nz_total / max(1, elts_total))
                        _k_eff_v = (k_eff_sum / denom) if denom > 0 else 0
                        try:
                            _dm = (sae.num_batches_not_active >= sae.config.get("n_batches_to_dead", 20))
                            _df = float(_dm.float().mean().item())
                        except Exception:
                            _df = 0.0
                        _rl2 = (rel_l2_sum / denom) if denom > 0 and rel_l2_sum != 0 else -1
                        _svar = (sum_svar / denom) if denom > 0 else 0
                        _bev_s = f" bev={_last_bev:.4f}" if _last_bev is not None else ""
                        _nhf_s = f" nhf={_last_nhf}" if _last_nhf is not None else ""
                        print(
                            f"[online] {lname} step={steps_done[lname]} "
                            f"loss={mean_loss:.4f} l2={mean_l2:.4f} aux={mean_aux:.4f} "
                            f"svar={_svar:.4f}{_bev_s}{_nhf_s} "
                            f"rel_l2={_rl2:.4f} sparsity={_sparsity:.6f} k_eff={_k_eff_v:.1f} "
                            f"dead_frac={_df:.4f}",
                            flush=True,
                        )

                    # === 히스토그램/노름 등 주기적 로깅 ===
                    if self.wandb_run and hist_every > 0 and steps_done[lname] % hist_every == 0:
                        try:
                            # 최근 마이크로배치의 활성 히스토그램 (샘플)
                            if 'fa' in locals() and fa is not None:
                                self._wandb_log({f"{safe}/acts_hist": wandb.Histogram(fa.detach().float().flatten().cpu().numpy())}, commit=False)
                            hist_payload = {
                                f"{safe}/W_enc_norm": float(sae.W_enc.detach().norm().item()),
                                f"{safe}/W_dec_norm": float(sae.W_dec.detach().norm().item()),
                            }
                            # Feature frequency histogram
                            if hasattr(sae, "feature_freq_ema"):
                                freq_np = sae.feature_freq_ema.float().cpu().numpy()
                                hist_payload[f"{safe}/freq_hist"] = wandb.Histogram(freq_np)
                            self._wandb_log(hist_payload)
                        except Exception:
                            pass


                    if self.wandb_run and hist_every > 0 and steps_done[lname] % hist_every == 0:
                        try:
                            self._wandb_log({
                                f"{safe}/W_enc_norm": float(sae.W_enc.detach().norm().item()),
                                f"{safe}/W_dec_norm": float(sae.W_dec.detach().norm().item()),
                            }, commit=False)
                            self._wandb_log({}, commit=True)  # flush
                        except Exception:
                            pass

                    # periodic memory log
                    mem_every = int(tr_cfg.get("mem_log_every_steps", 0))
                    if mem_every > 0 and (steps_done[lname] % mem_every == 0):
                        self._log_gpu_memory(f"train_step_{steps_done[lname]}")

                    # === ckpt 저장 ===
                    if ckpt_every > 0 and (steps_done[lname] % ckpt_every == 0):
                        self._save_layer_ckpt(lname, steps_done[lname], opt, self.tokens_seen[lname])
                        
            # 종료 조건: 모든 레이어 목표 달성(전 랭크 합) — collect 전에 체크
            local_remaining = sum(1 for ln in self.my_layers_to_train if steps_done[ln] < steps_goal)
            if dist.is_initialized():
                rem = torch.tensor([local_remaining], device=self.device, dtype=torch.long)
                dist.all_reduce(rem, op=dist.ReduceOp.SUM)
                all_remaining = int(rem.item())
            else:
                all_remaining = local_remaining

            if all_remaining == 0:
                if self.rank == 0:
                    logger.info("All layers reached target steps. Training finished.")
                break

            # 모든 랭크 학습 진행 여부 공유
            flag = torch.tensor([1.0 if any_trained_local else 0.0], device=self.device)
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            any_trained_global = flag.item() > 0.0

            # 학습이 전혀 안 되었으면 새로 수집
            if not any_trained_global:
                if dist.is_initialized():
                    self._barrier()
                if self.rank == 0:
                    logger.info(f"[step?] entering collect_round at global_step={global_step}")
                self.activation_store.collect_round()
                if self.rank == 0:
                    logger.info(f"[step?] finished collect_round at global_step={global_step}")
                if dist.is_initialized():
                    self._barrier()
                self._print_buffer_progress()

            # validation 주기 체크
            global_step = max(steps_done.values()) if steps_done else 0
            if val_every > 0 and (global_step - last_val_logged) >= val_every and self.validation_ready:
                self._run_validation(steps_done)
                last_val_logged = global_step


    def cleanup(self):
        if self.activation_store:
            self.activation_store.cleanup()


# ---------------------------- DDP launcher ----------------------------------

def setup(rank, world_size):
    # ✅ LOCAL_RANK 기준으로 현재 디바이스 먼저 지정
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    from datetime import timedelta
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=timedelta(hours=2),
    )
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def main_worker(rank, world_size, args):
    local_rank = setup(rank, world_size)   # ✅ 반환값 받기
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    pipe = SAETrainingPipeline(args.config, rank, world_size)
    try:
        pipe.train()
    finally:
        pipe.cleanup()
        cleanup_ddp()


__all__ = [
    "SAETrainingPipeline",
    "setup",
    "cleanup_ddp",
    "main_worker",
]
