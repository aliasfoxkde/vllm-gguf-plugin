# Qwen3.6 REAP Integration Progress

## Goal
Get `JZC973/Qwen3.6-35B-REAP-MTP-UD-GGUF-Collection/Qwen3.6-35B-A3B-UD-Q3_K_XL-REAP.gguf`
working with vLLM via this plugin.

## Status: Core config loading works, VllmConfig validation is the blocker

### What works

1. **GGUFConfigParser** reads REAP GGUF metadata directly via `load_gguf_checkpoint`
   (bypassing parent-dir config.json). Returns correct config:
   - `model_type: qwen3_5_moe_text`
   - `num_hidden_layers: 40` (block_count=41, nextn=1)
   - `num_experts: 192`, `num_experts_per_tok: 8`
   - `vocab_size: 248320`
   - `architectures: ['Qwen3_5MoeForCausalLM']`
   - `mtp_num_hidden_layers: 1`

2. **Plugin registration** works: `vllm_gguf_plugin.register()` is called
   via `vllm.general_plugins` entry point. GGUF loader, quant config,
   and MTP speculative decoding all register OK.

3. **Model registration** works: `Qwen3_5MoeForCausalLM` is registered in
   `vllm.ModelRegistry`.

### What's broken

`VllmConfig(...)` construction (called by `create_engine_config`) raises:
```
ValueError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
/home/mkinney/Models/JZC973/Qwen3.6-35B-REAP-MTP-UD-GGUF-Collection/
Qwen3.6-35B-A3B-UD-Q3_K_XL-REAP.gguf
```

This happens **after** `ModelConfig` is created successfully. The error
originates from a Pydantic validator inside `VllmConfig.__init__`, NOT from
`get_hf_image_processor_config`. All three image-processor patches are working
correctly (verified with debug output showing `is_gguf=True` and `returning empty`).

### Root cause investigation

- `model_config.model` = GGUF file path
- `model_config.hf_config_path` = `Qwen/Qwen3.6-35B-A3B` (redirected)
- `VllmConfig.__init__` calls pydantic validators after all fields are set
- No `get_hf_image_processor_config` calls appear in debug output after
  the ModelConfig succeeds (first call, then second call in create_engine_config)
- The error is not from `SpeculativeConfig` (no speculative config passed)
- The error is NOT from `get_hf_image_processor_config` — it appears the patch is working

### Attempted fixes

1. **Set `hf_config_path = "Qwen/Qwen3.6-35B-A3B"`** (base HF repo) to redirect
   image-processor lookup away from the .gguf file. This changes the error
   path but the same ValueError persists from within `VllmConfig.__init__`.
   The actual repo ID being validated is the HF redirect, not the GGUF file.

2. **Patched `get_hf_image_processor_config` at 3 layers**: this works
   (confirmed with debug) but doesn't solve the underlying VllmConfig error.

3. **`hf_config_path` redirect in `_patch_engine_args()`**: sets
   `hf_config_path = "Qwen/Qwen3.6-35B-A3B"` so image processor lookup goes
   to the real HF repo. But `ModelConfig.__post_init__` still uses `self.model`
   (the GGUF path) for other lookups.

### Key code locations

- `vllm_gguf_plugin/plugin.py:52` — `_patch_engine_args()`: sets `hf_config_path`
  to base HF repo for GGUF files
- `vllm_gguf_plugin/config_parser.py:53` — `GGUFConfigParser._parse_gguf_file()`:
  reads GGUF directly via `load_gguf_checkpoint`, sets architectures
- `vllm_gguf_plugin/qwen35_config.py:50` — `map_qwen35_config()`: pure function
  that builds Qwen3_5MoeTextConfig from GGUF metadata
- `vllm_gguf_plugin/plugin.py:120` — `_patch_hf_image_processor()`: patches
  image processor config to return `{}` for GGUF files (3 layers)

### Model specs (REAP GGUF)

```
Architecture: qwen35moe
block_count: 41 (40 layers + 1 MTP)
nextn_predict_layers: 1
num_experts: 192 (ATBender) / 180 (RangerX)
num_experts_per_tok: 8
hidden_size: 2048
intermediate_size: 0 (needs expert_feed_forward_length=512 from GGUF)
head_dim: 256
num_attention_heads: 16
num_key_value_heads: 2
Tensor types: Q6_K(4), F32(368), Q8_0(259), IQ4_XS(39), IQ3_XXS(78), Q4_K(1), Q3_K(2), BF16(2)
```

Layers 0–39: GDN (ssm_a, ssm_alpha, etc.)
Layer 40: Full attention + MTP head (blk.40.nextn.*)

### Next steps

1. **Debug VllmConfig validator**: Add debug prints to `VllmConfig.__init__`
   to identify which field's validator is failing
2. **Check if SpeculativeConfig is auto-created** even without `--spec-method`:
   `create_speculative_config()` may auto-create a config when `target_model_config`
   has MTP-related attributes
3. **Try with explicit `--hf-config-path` pointing to a real HF model dir**:
   `vllm serve <gguf> --hf-config-path ~/.cache/huggingface/modules/...`
4. **Check `VLLM_MODEL_REDIRECT_PATH`**: Could redirect the GGUF path to a real
   HF model directory containing only config.json

### DSpark integration (future work)

- DSpark GGUF: `ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF`
- DSpark = speculative decoding using a separate draft model
- vLLM supports `dspark` method in `SpeculativeConfig`
- Would need to implement GGUF tensor mapping for the draft model architecture
- Benchmark plan: concurrency [2,4,8,12,14,16,18,20,24] for MTP vs DSpark vs llama.cpp
