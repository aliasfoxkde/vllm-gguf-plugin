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
            # Don't redirect hf_config_path - keep it pointing to the GGUF file.
            # The GGUFConfigParser handles config reading from GGUF metadata.
            # Image processor config will be empty for text-only GGUF models.
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True


def _patch_get_quant_config() -> None:
    """Patch get_quant_config to bypass HF Hub downloads for GGUF files.

    GGUF quantization config is in the GGUF metadata, not in separate JSON
    files on HF Hub. Without this patch, get_quant_config calls
    hf_api().snapshot_download which fails on .gguf file paths.
    """
    if getattr(_patch_get_quant_config, "_patched", False):
        return

    from .gguf_utils import check_gguf_file as _check_gguf
    from .quantization.config import GGUFConfig

    import vllm.model_executor.model_loader.weight_utils as weight_utils
    original_get_quant_config = weight_utils.get_quant_config

    def _patched_get_quant_config(model_config, load_config):
        if _check_gguf(model_config.model):
            # GGUF files carry quantization config in metadata; no HF Hub download needed.
            # Return a GGUFConfig instance so the downstream quantization checks succeed.
            return GGUFConfig()
        return original_get_quant_config(model_config, load_config)

    weight_utils.get_quant_config = _patched_get_quant_config
    _patch_get_quant_config._patched = True  # type: ignore[attr-defined]


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
    """Patch HF-related functions for GGUF files.

    Four layers need patching because vLLM and transformers do `from ... import`
    at module load time, capturing local references:
    1. `vllm.transformers_utils.config.get_hf_image_processor_config` — top-level
    2. `vllm.config.model.get_hf_image_processor_config` — direct import in model.py
    3. `transformers.models.auto.image_processing_auto.get_image_processor_config`
       — called by vllm's wrapper
    4. `vllm.transformers_utils.repo_utils.try_get_local_file` — called from
       get_sentence_transformer_tokenizer_config; bypasses HF validation for GGUF

    Each patch returns {} or None for GGUF files, bypassing HF Hub lookups.
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

    # Layer 4: try_get_local_file — called from get_sentence_transformer_tokenizer_config
    # For GGUF files, return None immediately (sentence-transformer configs don't apply).
    # The function is imported directly into vllm.transformers_utils.config, so we must
    # patch the reference there (not just in repo_utils).
    try:
        import vllm.transformers_utils.repo_utils as repo_utils
        if not getattr(repo_utils.try_get_local_file, "_gguf_patched", False):
            _orig_try_get_local_file = repo_utils.try_get_local_file

            def _patched_try_get_local_file(model, file_name, revision=None):
                if _check_gguf(model):
                    return None
                return _orig_try_get_local_file(model, file_name, revision)

            _patched_try_get_local_file._gguf_patched = True  # type: ignore[attr-defined]
            repo_utils.try_get_local_file = _patched_try_get_local_file

        # Also patch the direct import in transformers_utils.config
        import vllm.transformers_utils.config as tu_config
        if not getattr(tu_config.try_get_local_file, "_gguf_patched", False):
            tu_config.try_get_local_file = repo_utils.try_get_local_file
    except Exception:
        pass


def _install_turboquant() -> None:
    """Install TurboQuant+ KV cache and/or weight compression if requested via env vars.

    Enabled by setting VLLM_GGUF_TURBOQUANT=1 and optionally:
    - VLLM_GGUF_TURBOQUANT_KV_K_BITS: Key bits (default 4)
    - VLLM_GGUF_TURBOQUANT_KV_V_BITS: Value bits (default 4)
    - VLLM_GGUF_TURBOQUANT_WEIGHT_BITS: Weight bits (default 3)
    - VLLM_GGUF_TURBOQUANT_PRUNE_EXPERTS: MoE expert pruning fraction (default 0)
    """
    import os as _os

    if not _os.environ.get("VLLM_GGUF_TURBOQUANT", "").strip():
        return

    try:
        from .turboquant_integration import install_turboquant
    except ImportError:
        return

    kv_k = int(_os.environ.get("VLLM_GGUF_TURBOQUANT_KV_K_BITS", "4"))
    kv_v = int(_os.environ.get("VLLM_GGUF_TURBOQUANT_KV_V_BITS", "4"))
    weight_bits = int(_os.environ.get("VLLM_GGUF_TURBOQUANT_WEIGHT_BITS", "3"))
    prune = float(_os.environ.get("VLLM_GGUF_TURBOQUANT_PRUNE_EXPERTS", "0"))

    result = install_turboquant(
        kv_k_bits=kv_k,
        kv_v_bits=kv_v,
        weight_bits=weight_bits,
        prune_experts=prune,
    )
    if result["kv"]:
        print(f"[vllm_gguf_plugin] TurboQuant KV compression installed (K={kv_k}, V={kv_v})")
    if result["weights"]:
        print(f"[vllm_gguf_plugin] TurboQuant weight compression installed ({weight_bits} bit)")


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
    _patch_get_quant_config()
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

    # TurboQuant+ KV cache and weight compression
    # Only install if explicitly requested via env vars
    _install_turboquant()
