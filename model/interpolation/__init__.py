"""Models registered for the interpolation task.

Importing this sub-package executes every concrete model file so their
``@register_model`` decorators run and populate the shared registry exposed by
``model.registry``.
"""

from ..registry import MODEL_REGISTRY, build_model, register_model

from . import atten_unet  # noqa: F401
from . import chai2020_unet  # noqa: F401
from . import dncnn  # noqa: F401
from . import res_unet  # noqa: F401
from . import unet  # noqa: F401
from . import unet_plusplus  # noqa: F401
from . import wang2019_resnet  # noqa: F401
from . import yoon2021_dbilstm  # noqa: F401
from . import li2022_caunet  # noqa: F401
from . import liu2022_wrdl  # noqa: F401
from . import pan2020_pconv_unet  # noqa: F401
from . import park2022_cfunet  # noqa: F401
from . import yu2022_anet  # noqa: F401
from . import yuan2022_btn  # noqa: F401
from . import guo2023_mst  # noqa: F401
from . import wu2024_spnet  # noqa: F401
from . import gated_transformer_v9  # noqa: F401
from . import gated_transformer_v9_encdec  # noqa: F401
from . import gated_transformer_v11  # noqa: F401
from . import gated_transformer_v11_encdec  # noqa: F401
from . import trace_token_transformer_interpolator  # noqa: F401
from . import hf_vit_interpolator  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
