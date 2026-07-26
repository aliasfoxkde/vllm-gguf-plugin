# SPDX-License-Identifier: Apache-2.0

from .op_guard import install_idempotent_op_registration

# Must run before .quantization (and before vllm core's quantization/gguf.py)
# imports register duplicate vllm:: custom ops — see op_guard.py.
install_idempotent_op_registration()

try:
    from .offload_instrument import install as _install_offload_instrument

    _install_offload_instrument()
except ImportError:
    pass

# Must patch before any vLLM module is imported.
# This is why it's here in __init__.py rather than in register().
from .plugin import _patch_hf_image_processor

_patch_hf_image_processor()

from .config_parser import GGUFConfigParser  # noqa: E402
from .loader import GGUFModelLoader
from .plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from .quantization import DiffusionGGUFConfig, GGUFConfig

__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
