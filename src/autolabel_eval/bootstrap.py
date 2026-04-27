from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from .config import EvalConfig


def bootstrap_speclens(config: EvalConfig) -> None:
    repo_root = str(config.repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    src_path = config.repo_root / "src"
    existing = sys.modules.get("src")
    if existing is not None and str(src_path) in list(getattr(existing, "__path__", [])):
        return
    spec = importlib.util.spec_from_file_location(
        "src",
        str(src_path / "__init__.py"),
        submodule_search_locations=[str(src_path)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to bootstrap src from {src_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["src"] = module
    spec.loader.exec_module(module)


def load_module_from_path(module_name: str, file_path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def register_research_saes(config: EvalConfig) -> None:
    bootstrap_speclens(config)
    module_name = "autolabel_research_variants"
    if module_name in sys.modules:
        return
    module = load_module_from_path(module_name, config.research_variants_py)
    register = getattr(module, "register_research_saes", None)
    if callable(register):
        register()

