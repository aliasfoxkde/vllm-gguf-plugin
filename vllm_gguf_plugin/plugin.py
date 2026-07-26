# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from pathlib import Path

import vllm.engine.arg_utils as arg_utils_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .config_parser import GGUFConfigParser
from .gguf_utils import check_gguf_file, is_gguf, is_remote_gguf, split_remote_gguf
from .loader import GGUFModelLoader
from .quantization import DiffusionGGUFConfig, GGUFConfig
from .qwen35_config import register as _register_qwen35_gguf
from .weights_adapter.diffusion.integration import _patch_diffusers_loader

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _get_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    if is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        return repo_id
    if check_gguf_file(model):
        return str(Path(model).parent)
    return model


def _patch_engine_args() -> None:
    if getattr(EngineArgs, "_gguf_create_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_model
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
            self.model = _get_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
            # hf_config_path must point to the GGUF file (not its parent dir)
            # so GGUFConfigParser can parse it directly via load_gguf_checkpoint.
            # ModelConfig uses hf_config_path or model (whichever is set) to load
            # the config - keeping model=gguf_path is fine with config_format=gguf.
            if self.hf_config_path is None:
                self.hf_config_path = gguf_model
            # For standalone GGUF files (no config.json sibling), keep model pointing
            # at the GGUF file so get_config parses it via GGUFConfigParser directly.
            if self.model == gguf_model or self.model != self.hf_config_path:
                self.model = gguf_model
            # Redirect hf_config_path to the Qwen3.6 base HF repo so
            # get_hf_image_processor_config (which calls HF Hub) doesn't
            # try to validate the .gguf path as a repo ID. GGUFConfigParser
            # handles the actual config reading from the GGUF file directly.
            if self.hf_config_path == gguf_model:
                self.hf_config_path = "Qwen/Qwen3.6-35B-A3B"
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True


def _patch_speculator_probe() -> None:
    if getattr(arg_utils_module, "_gguf_speculator_probe_patched", False):
        return

    original_maybe_override = arg_utils_module.maybe_override_with_speculators

    @wraps(original_maybe_override)
    def maybe_override_with_speculators(model, tokenizer, *args, **kwargs):
        if _is_gguf_reference(model):
            return model, tokenizer, kwargs.get("vllm_speculative_config")
        return original_maybe_override(model, tokenizer, *args, **kwargs)

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def _register_omni_diffusion_quantization() -> None:
    try:
        from vllm_omni.quantization import register_quantization_override
    except ImportError:
        return

    register_quantization_override("gguf", lambda **kw: DiffusionGGUFConfig(**kw))


def _patch_hf_image_processor() -> None:
    """Patch get_hf_image_processor_config for GGUF files.

    Three layers need patching because vLLM and transformers do `from ... import`
    at module load time, capturing local references:
    1. `vllm.transformers_utils.config.get_hf_image_processor_config` — top-level
    2. `vllm.config.model.get_hf_image_processor_config` — direct import in model.py
    3. `transformers.models.auto.image_processing_auto.get_image_processor_config`
       — called by vllm's wrapper

    Each patch returns {} for GGUF files, bypassing HF Hub image-processor lookup.
    """
    from .gguf_utils import check_gguf_file as _check_gguf
    _empty = {}

    def _make_patcher(name, orig_fn):
        # Don't re-patch an already-patched function
        if getattr(orig_fn, "_gguf_patched", False):
            return orig_fn

        def _patched(model, **kwargs):
            if _check_gguf(model):
                return _empty
            return orig_fn(model, **kwargs)

        _patched._gguf_patched = True  # type: ignore[attr-defined]
        return _patched

    # Layer 1: vllm.transformers_utils.config
    try:
        import vllm.transformers_utils.config as tu_config
        tu_config.get_hf_image_processor_config = _make_patcher(
            "vllm.transformers_utils.config",
            tu_config.get_hf_image_processor_config,
        )
    except Exception:
        pass

    # Layer 2: vllm.config.model
    try:
        from vllm.config import model as model_module
        if not getattr(model_module.get_hf_image_processor_config, "_gguf_patched", False):
            model_module.get_hf_image_processor_config = _make_patcher(
                "vllm.config.model",
                model_module.get_hf_image_processor_config,
            )
    except Exception:
        pass

    # Layer 3: transformers.models.auto.image_processing_auto
    try:
        import transformers.models.auto.image_processing_auto as ip_auto
        ip_auto.get_image_processor_config = _make_patcher(
            "transformers.auto",
            ip_auto.get_image_processor_config,
        )
    except Exception:
        pass


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    register_quantization_config("gguf")(GGUFConfig)
    _register_omni_diffusion_quantization()

    if "gguf" not in _LOAD_FORMAT_TO_MODEL_LOADER or not isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader
    ):
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    _patch_engine_args()
    _patch_speculator_probe()
    _patch_hf_image_processor()
    _patch_diffusers_loader()
    _register_qwen35_gguf()
    from .mtp_enable import install as _install_mtp

    _install_mtp()

    from .mtp_dynamic_skip import install as _install_mtp_dynamic_skip

    _install_mtp_dynamic_skip()

    from .spec_plan_cache import install as _install_spec_plan_cache

    _install_spec_plan_cache()

    from .mtp_eagle_group import _patch_eagle_group_annotation

    _patch_eagle_group_annotation()

    from .sleep_wake_hybrid import _patch_init_fp8_kv_scales

    _patch_init_fp8_kv_scales()
