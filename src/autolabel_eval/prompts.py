from __future__ import annotations

from .bootstrap import load_module_from_path
from .config import EvalConfig


def load_baseline_prompt(config: EvalConfig) -> dict[str, str]:
    module = load_module_from_path(
        "autolabel_baseline_prompt_module",
        config.baseline_repo_root / "sae_auto_interp/agents/explainers/image_explainer/prompts.py",
    )
    return {
        "system_prompt": str(getattr(module, "SYSTEM")),
        "user_guidelines": str(getattr(module, "GUIDELINES")),
        "prompt_version": config.prompt_version_baseline,
    }


def pairwise_prompt(config: EvalConfig) -> dict[str, str]:
    system = (
        "You evaluate whether a candidate SAE feature label is active in one target vision token. "
        "Use only the provided token-centered evidence. Do not judge whole-image semantics."
    )
    user = (
        "You will receive three images for one token: "
        "(1) the original image with the target token boxed, "
        "(2) a binary token ERF support image showing the minimal sufficient support set, and "
        "(3) a token-neighbor cosine map showing contextual similarity to other tokens. "
        "You will also receive one candidate feature label. "
        "Return JSON with keys score_0_100 (integer 0-100), decision ('yes' or 'no'), and brief_reason. "
        "Score should reflect confidence that the candidate feature is active at the held-out token."
    )
    return {
        "system_prompt": system,
        "user_guidelines": user,
        "prompt_version": config.prompt_version_pairwise,
    }


def axis1_c_conditioned_prompt(config: EvalConfig) -> dict[str, str]:
    system = (
        "You evaluate whether one fixed SAE feature label is active in one target vision token. "
        "The source image is known to contain the feature somewhere, so you must judge the boxed token itself."
    )
    user = (
        "You will receive three token-centered evidence images: "
        "(1) the original image with the target token boxed, "
        "(2) a binary token ERF support image for that token, and "
        "(3) a token-neighbor cosine map for that token. "
        "You will also receive one feature label that is known to be present somewhere in the image. "
        "Decide whether the boxed token itself expresses that feature. "
        "Return JSON with keys score_0_100 (integer 0-100), decision ('yes' or 'no'), and brief_reason."
    )
    return {
        "system_prompt": system,
        "user_guidelines": user,
        "prompt_version": config.prompt_version_axis1_c_conditioned,
    }
