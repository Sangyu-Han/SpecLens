N_PATCHES = 196   # 14×14 patches for ViT-B/16 at 224px
PATCH_DIM = 768


def tok_to_yx(tok_idx: int) -> tuple:
    """Flat patch token index (1-based, 1..196) → (row, col) in 14×14 grid."""
    return divmod(tok_idx - 1, 14)
