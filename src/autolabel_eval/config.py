from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


DEFAULT_REPO_ROOT = Path("/home/sangyu/Desktop/Master/SpecLens")
DEFAULT_CLIP50K_INDEX_ROOT = DEFAULT_REPO_ROOT / "outputs/spec_lens_store/clip_50k_index"
DEFAULT_CLIP50K_SAE_ROOT = DEFAULT_REPO_ROOT / "outputs/spec_lens_store/clip_50k_sae"
DEFAULT_CLIP50K_MODEL_NAME = "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"
DEFAULT_LEGACY_AUTOLABEL_ROOT = Path("/home/sangyu/Desktop/Master/speclens_trash/research_sae_autolabel")
DEFAULT_LEGACY_BIAS_ROOT = Path("/home/sangyu/Desktop/Master/speclens_trash/claude_research_bias")
DEFAULT_LEGACY_VARIANT_ROOT = Path("/home/sangyu/Desktop/Master/speclens_trash/codex_research_bias")


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class EvalConfig:
    workspace_root: Path = Path("/home/sangyu/Desktop/Master/codex_research_autolabeling")
    repo_root: Path = DEFAULT_REPO_ROOT
    # Legacy roots are kept only for remapping stale checkpoint-internal paths.
    legacy_autolabel_root: Path = DEFAULT_LEGACY_AUTOLABEL_ROOT
    legacy_bias_root: Path = DEFAULT_LEGACY_BIAS_ROOT
    legacy_variant_root: Path = DEFAULT_LEGACY_VARIANT_ROOT
    baseline_repo_root: Path = Path("/home/sangyu/Desktop/Master/SpecLens/third_party/multimodal-sae-main")
    device: str = field(default_factory=_default_device)
    model_name: str = DEFAULT_CLIP50K_MODEL_NAME
    blocks: tuple[int, ...] = (2, 6, 10)
    features_per_block: int = 64
    train_examples_per_feature: int = 5
    holdout_examples_per_feature: int = 2
    min_mean_score: float = 5.0
    random_seed: int = 42
    shuffle_feature_candidates: bool = True
    image_size: int = 224
    resize_size: int = 256
    grid_size: int = 14
    n_patches: int = 196
    target_ratio_min: float = 0.30
    min_feature_max_act: float = 0.50
    p99_percentile: float = 99.0
    cautious_steps: int = 32
    cautious_lr: float = 0.45
    cautious_lr_end: float = 0.01
    cautious_tv_weight: float = 0.01
    cautious_irr_weight: float = 0.05
    cautious_init_prob: float = 0.50
    cautious_seed: int = 0
    erf_objective_mode: str = "cosine"
    erf_recovery_threshold: float = 0.80
    erf_support_min_normalized_attribution: float = 0.10
    decision_threshold: float = 0.50
    prompt_version_baseline: str = "multimodal_sae_image_explainer_local_v1"
    prompt_version_pairwise: str = "token_pairwise_eval_v2"
    prompt_version_axis1_c_conditioned: str = "token_c_conditioned_eval_v2"
    axis1_positive_mode: str = "target_only"
    axis1_negatives_per_image: int = 4
    axis1_negative_strategy: str = "lowest_activation"
    human_session_default_features_per_block: int = 4
    human_session_default_seed: int = 0
    study_session_default_features_per_block: int = 2
    study_session_default_seed: int = 0
    autolabel_session_default_features_per_block: int = 2
    autolabel_session_default_seed: int = 0
    study_label_examples_per_feature: int = 10
    autolabel_label_examples_by_block: dict[int, int] = field(
        default_factory=lambda: {
            2: 18,
            6: 18,
            10: 10,
        }
    )
    study_axis1_easy_negatives_per_holdout: int = 1
    study_axis1_hard_negatives_per_holdout: int = 1
    study_axis2_candidates_per_item: int = 4
    axis2_positive_relative_threshold: float = 0.50
    axis2_metric_ks: tuple[int, ...] = (1, 3, 5)
    exclude_feature_ids: dict[int, list[int]] = field(
        default_factory=lambda: {
            2: [603, 6561, 9395, 9537, 13045, 14601],
            6: [9167, 9909, 12105, 12786, 12891, 16521, 16857],
            10: [6, 19359],
        }
    )
    deciles_root_override: Path | None = None
    offline_meta_root_override: Path | None = None
    checkpoints_root_override: Path | None = None
    checkpoint_relpath_template: str = "model.blocks.{block_idx}/step_0050000_tokens_204800000.pt"
    dataset_root_override: Path | None = None

    @property
    def deciles_root(self) -> Path:
        if self.deciles_root_override is not None:
            return Path(self.deciles_root_override)
        return DEFAULT_CLIP50K_INDEX_ROOT / "deciles"

    @property
    def offline_meta_root(self) -> Path:
        if self.offline_meta_root_override is not None:
            return Path(self.offline_meta_root_override)
        return DEFAULT_CLIP50K_INDEX_ROOT / "offline_meta"

    @property
    def checkpoints_root(self) -> Path:
        if self.checkpoints_root_override is not None:
            return Path(self.checkpoints_root_override)
        return DEFAULT_CLIP50K_SAE_ROOT

    @property
    def research_variants_py(self) -> Path:
        return self.legacy_variant_root / "src/variants.py"

    @property
    def feature_bank_json(self) -> Path:
        return self.workspace_root / "outputs/feature_bank.json"

    @property
    def baseline_label_requests_json(self) -> Path:
        return self.workspace_root / "outputs/baseline_label_requests.json"

    @property
    def baseline_labels_json(self) -> Path:
        return self.workspace_root / "outputs/baseline_labels.json"

    @property
    def pairwise_requests_jsonl(self) -> Path:
        return self.workspace_root / "outputs/pairwise_judge_requests.jsonl"

    @property
    def pairwise_scores_jsonl(self) -> Path:
        return self.workspace_root / "outputs/pairwise_scores.jsonl"

    @property
    def metrics_summary_json(self) -> Path:
        return self.workspace_root / "outputs/metrics_summary.json"

    @property
    def ground_truth_json(self) -> Path:
        return self.workspace_root / "outputs/ground_truth_summary.json"

    @property
    def token_evidence_json(self) -> Path:
        return self.workspace_root / "outputs/token_evidence.json"

    @property
    def axis1_dataset_json(self) -> Path:
        return self.workspace_root / "outputs/axis1_c_conditioned_eval.json"

    @property
    def axis1_evidence_json(self) -> Path:
        return self.workspace_root / "outputs/axis1_c_conditioned_evidence.json"

    @property
    def axis1_requests_jsonl(self) -> Path:
        return self.workspace_root / "outputs/axis1_c_conditioned_requests.jsonl"

    @property
    def axis1_scores_jsonl(self) -> Path:
        return self.workspace_root / "outputs/axis1_c_conditioned_scores.jsonl"

    @property
    def axis1_metrics_json(self) -> Path:
        return self.workspace_root / "outputs/axis1_c_conditioned_metrics.json"

    @property
    def axis_feature_subsets_json(self) -> Path:
        return self.workspace_root / "outputs/axis_feature_subsets.json"

    @property
    def axis_label_demand_root(self) -> Path:
        return self.workspace_root / "outputs/axis_label_demand"

    @property
    def axis_label_sessions_root(self) -> Path:
        return self.workspace_root / "outputs/axis_label_sessions"

    @property
    def axis_runs_root(self) -> Path:
        return self.workspace_root / "outputs/axis_runs"

    @property
    def axis_audits_root(self) -> Path:
        return self.workspace_root / "outputs/axis_audits"

    @property
    def human_study_root(self) -> Path:
        return self.workspace_root / "outputs/human_study"

    @property
    def study_root(self) -> Path:
        return self.workspace_root / "outputs/study_sessions"

    @property
    def label_registry_jsonl(self) -> Path:
        return self.workspace_root / "outputs/label_registry.jsonl"

    @property
    def autolabel_root(self) -> Path:
        return self.workspace_root / "outputs/autolabel_sessions"

    @property
    def evidence_root(self) -> Path:
        return self.workspace_root / "outputs/evidence"

    @property
    def diagnostics_root(self) -> Path:
        return self.workspace_root / "outputs/diagnostics"

    @property
    def axis_experiments_root(self) -> Path:
        return self.workspace_root / "outputs/axis_experiments"

    @property
    def cache_root(self) -> Path:
        return self.workspace_root / "cache"

    def checkpoint_path(self, block_idx: int) -> Path:
        relpath = str(self.checkpoint_relpath_template).format(block_idx=int(block_idx))
        return self.checkpoints_root / relpath

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["deciles_root"] = str(self.deciles_root)
        payload["offline_meta_root"] = str(self.offline_meta_root)
        payload["checkpoints_root"] = str(self.checkpoints_root)
        payload["research_variants_py"] = str(self.research_variants_py)
        payload["feature_bank_json"] = str(self.feature_bank_json)
        payload["baseline_label_requests_json"] = str(self.baseline_label_requests_json)
        payload["baseline_labels_json"] = str(self.baseline_labels_json)
        payload["pairwise_requests_jsonl"] = str(self.pairwise_requests_jsonl)
        payload["pairwise_scores_jsonl"] = str(self.pairwise_scores_jsonl)
        payload["metrics_summary_json"] = str(self.metrics_summary_json)
        payload["ground_truth_json"] = str(self.ground_truth_json)
        payload["token_evidence_json"] = str(self.token_evidence_json)
        payload["axis1_dataset_json"] = str(self.axis1_dataset_json)
        payload["axis1_evidence_json"] = str(self.axis1_evidence_json)
        payload["axis1_requests_jsonl"] = str(self.axis1_requests_jsonl)
        payload["axis1_scores_jsonl"] = str(self.axis1_scores_jsonl)
        payload["axis1_metrics_json"] = str(self.axis1_metrics_json)
        payload["axis_feature_subsets_json"] = str(self.axis_feature_subsets_json)
        payload["axis_label_demand_root"] = str(self.axis_label_demand_root)
        payload["axis_label_sessions_root"] = str(self.axis_label_sessions_root)
        payload["axis_runs_root"] = str(self.axis_runs_root)
        payload["axis_audits_root"] = str(self.axis_audits_root)
        payload["study_root"] = str(self.study_root)
        payload["autolabel_root"] = str(self.autolabel_root)
        payload["label_registry_jsonl"] = str(self.label_registry_jsonl)
        payload["axis_experiments_root"] = str(self.axis_experiments_root)
        return payload

    def ensure_dirs(self) -> None:
        for path in (
            self.workspace_root,
            self.workspace_root / "outputs",
            self.workspace_root / "outputs/requests",
            self.study_root,
            self.autolabel_root,
            self.evidence_root,
            self.diagnostics_root,
            self.axis_label_demand_root,
            self.axis_label_sessions_root,
            self.axis_runs_root,
            self.axis_audits_root,
            self.axis_experiments_root,
            self.cache_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> EvalConfig:
    cfg = EvalConfig()
    cfg.ensure_dirs()
    return cfg
