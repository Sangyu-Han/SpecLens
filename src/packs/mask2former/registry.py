# src/packs/mask2former/registry.py
"""Registry for Mask2Former pack components."""

REGISTRY = {
    # Model loader
    "model_loader": "src.packs.mask2former.models.model_loaders:load_mask2former_hf",
    
    # Dataset builders (for SA-V dataset)
    "dataset_builder": "src.packs.mask2former.dataset.builders:build_indexing_dataset",
    "collate_builder": "src.packs.mask2former.dataset.builders:build_collate_fn",
    
    # Store factory (adapter-based)
    "store_factory": "src.packs.mask2former.models.adapters:create_mask2former_store",
}
