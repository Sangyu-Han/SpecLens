from omegaconf import OmegaConf
from hydra.utils import instantiate
from training.utils import checkpoint_utils
import torch

# -------------------- Model loader --------------------
def load_sam2(cfg_m, device, logger=None):
    hydra_cfg = OmegaConf.load(cfg_m["yaml"])
    if "trainer" in hydra_cfg and "model" in hydra_cfg.trainer:
        model_cfg = hydra_cfg.trainer.model
    elif "model" in hydra_cfg:
        model_cfg = hydra_cfg.model
    else:
        raise KeyError("YAML에 model 블록이 없습니다.")
    model = instantiate(model_cfg).to(device).eval()
    # ckpt
    ckpt_path = cfg_m.get("ckpt")
    if ckpt_path is None:
        ckpt_path = hydra_cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path
    try:
        weights = checkpoint_utils.load_checkpoint_and_apply_kernels(checkpoint_path=str(ckpt_path),
                                                                     ckpt_state_dict_keys=["model"])
        state_dict = weights.get("model", weights)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "model" in raw:
            state_dict = raw["model"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]; state_dict = {k.replace("module.","",1): v for k,v in sd.items()}
        else:
            state_dict = raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    try:
        (f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    except:
        pass
    return model