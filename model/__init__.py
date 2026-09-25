"""Public API for the ``model`` package.

Each subtask folder (``model/<task>/``) is a self-contained sub-package: its
``__init__.py`` imports its concrete model files so their ``@register_model``
decorators populate the shared registry. To use a subtask's models, import
``build_model`` from that subtask, e.g.::

    from model.ground_roll_attenuation import build_model

or::

    from model.multiples_attenuation import build_model

This top-level package only exposes the registry primitives plus the
identity-forward ``placeholder`` so generic tooling can run without picking a
subtask. Loading multiple subtasks in the same Python process is not supported
because they register the same model names (``unet``, ``res_unet``, ...).
"""

from .registry import MODEL_REGISTRY, build_model, register_model
from . import placeholder_model  # noqa: F401  (registers ``placeholder``)

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
