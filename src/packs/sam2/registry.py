# src/packs/sam2/registry.py
REGISTRY = {
    # 모델/데이터셋/스토어
    "model_loader": "src.packs.sam2.models.loader:load_sam2",
    "dataset_builder": "src.packs.sam2.dataset.sa_v.builders:build_indexing_dataset",
    "collate_builder": "src.packs.sam2.dataset.sa_v.repro_vosdataset:make_collate",

    # SAM2 전용 스토어 팩토리(어댑터 생성 포함)
    #  - 어댑터/팩토리를 같은 파일에 두셨다면 adapters:create_sam2eval_store 로 두세요.
    #  - 분리하셨다면 factory:create_sam2eval_store 로 맞추세요.
    "store_factory": "src.packs.sam2.models.adapters:create_sam2eval_store",

    # 오프라인 메타 레저(SAM2 전용: prompts 포함) — 기존 offline_meta_ledger.py 이동본
    "offline_meta_ledger": "src.packs.sam2.offline.ledger:OfflineMetaParquetLedger",

    # BVD 재조립 유틸(검증/리컨스트럭트에서 사용)
    "bvd_builder_with_prompt": "src.packs.sam2.offline.bvd_builders:make_single_bvd_with_prompt",
    "bvd_builder_no_prompt": "src.packs.sam2.offline.bvd_builders:make_single_bvd_no_prompt",
}
