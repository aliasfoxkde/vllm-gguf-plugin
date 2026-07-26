# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from .qwen35_config import _CONFIG_CLASS as QWEN35_CONFIG_CLASS, _read_gguf_metadata


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model

        # For local GGUF files: read config directly from GGUF metadata
        # (avoids needing a config.json in the parent directory)
        if check_gguf_file(model):
            return self._parse_gguf_file(model)

        resolved_model = self._resolve_config_source(model)
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config_dict["architectures"] = [model_type]
        config.update({"architectures": [model_type]})

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config

    def _parse_gguf_file(
        self, model: str | Path
    ) -> tuple[dict, PretrainedConfig]:
        """Parse a local GGUF file directly into a PretrainedConfig.

        Uses the wrapped load_gguf_checkpoint (via qwen35_config's monkey-patch)
        for qwen35/qwen35moe arches, falling back to the stock loader for others.
        """
        model = str(model)
        try:
            result = _gguf_utils.load_gguf_checkpoint(model, return_tensors=False)
            config = result["config"]
            # Ensure architectures is set (required by vLLM model registry)
            if config.architectures is None:
                model_type = getattr(config, "model_type", None) or ""
                arch = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type)
                if arch:
                    config.update({"architectures": [arch]})
        except Exception:
            # Fallback for non-qwen35 GGUF files: delegate to HuggingFace
            parent = str(Path(model).parent)
            config_dict, config = HFConfigParser().parse(
                parent,
                trust_remote_code=True,
            )
            config = maybe_patch_hf_config_from_gguf(model, config)
            return {}, config

        config_dict = config.to_dict()
        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
