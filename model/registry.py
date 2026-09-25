"""Model registry + factory. See ``model/README.md`` for the workflow."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Class decorator that registers a model under ``name``."""

    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate a model from a ``{type, params}`` config block."""
    name = cfg["type"]
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**cfg.get("params", {}))
