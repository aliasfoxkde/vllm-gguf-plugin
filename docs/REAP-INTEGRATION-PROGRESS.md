# Qwen3.6 REAP + TurboQuant Integration Progress

## Goal
Get `JZC973/Qwen3.6-35B-REAP-MTP-UD-GGUF-Collection` working with vLLM via this plugin, with TurboQuant KV cache compression and DSpark speculative decoding.

## Status (Aug 4, 2026): Config loading works, model loading OOM on 16GB GPU

### What works

1. **GGUFConfigParser** reads REAP GGUF metadata directly via `load_gguf_checkpoint`
   - Correctly identifies `Qwen3_5MoeForCausalLM` architecture
   - Returns correct `num_hidden_layers: 40` (block_count=41, nextn=1)
   - `num_experts: 180` (RangerX), `num_experts_per_tok: 8`

2. **VllmConfig validation errors FIXED** (commit dd66a4d):
   - `_patch_get_quant_config()` bypasses HF Hub downloads for GGUF files
   - `try_get_local_file` patched for sentence-transformer config lookup
   - Returns proper `GGUFConfig()` instead of None

3. **TurboQuant+ integration** (commit dd66a4d):
   - `turboquant_integration.py` module added
   - Enabled via `VLLM_GGUF_TURBOQUANT=1` env var
   - KV: K=4, V=4 bits via WHT rotation + Lloyd-Max
   - Weight: kurtosis-aware 3-bit compression for MoE
   - Expert pruning support (REAP-style)

### What's broken

**OOM during model initialization** on RTX 5060 Ti 16GB:
- Model file: `Qwen3.6-35B-A3B-UD-IQ3_XXS-REAP-RangerX.gguf` (~10.5GB on disk)
- vLLM allocates ~22GB RSS during model init (OOM-killed by kernel)
- Happens at "Enabled custom fusions: norm_quant, act_quant" step
- Even with `--enforce-eager`, `--gpu-memory-utilization 0.1`, `--max-model-len 128`
- CUDA graphs capture not involved (disabled in eager mode)

**Root cause**: vLLM's model initialization overhead exceeds available VRAM even for a quantized model.

### Research findings

1. **TurboQuant+ (varjoranta/turboquant-vllm)**:
   - Explicitly supports Qwen3.6-35B-A3B MoE + partial rotary
   - Supports MoE weight compression with kurtosis-aware bit allocation
   - KV compression via WHT rotation + Lloyd-Max quantization
   - Compatible with vLLM 0.19-0.20 and 0.25.x

2. **DSpark speculative decoding**:
   - Method: `dspark` in `SpeculativeConfig`
   - Automatically detected if "dspark" in draft model name
   - Draft model: `ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF` (~1.5GB)
   - Compatible with Qwen3 model family
   - Supports heterogeneous vocab

3. **DFlash + KV quantization incompatibility**:
   - DFlash requires `causal=False` but turboquant backend hardcodes `causal=True`
   - Issue: https://github.com/vllm-project/vllm/issues/41559
   - DSpark likely doesn't have this issue (not confirmed)

4. **TurboQuant 4-bit variants**:
   - `turboquant_4bit_nc` = 3.4x KV cache reduction, modest accuracy loss
   - `turboquant_k8v4` = 2.4x reduction, minimal advantage over FP8
   - Avoid `k3v4-nc` and `3bit-nc` (up to 20 accuracy points drop)

### Model specs

| Model | Size | Experts | Quantization |
|-------|------|---------|-------------|
| Qwen3.6-35B-A3B-UD-IQ3_XXS-REAP | ~10.5GB | 192 (ATBender) | IQ3_XXS |
| Qwen3.6-35B-A3B-UD-IQ3_XXS-REAP-RangerX | ~10.5GB | 180 (RangerX) | IQ3_XXS |
| Qwen3.6-35B-A3B-UD-Q3_K_XL-REAP | ~13.4GB | 192 (ATBender) | Q3_K_XL |
| ankk98/dspark-qwen3-8b-block7-Q4_K_M | ~1.5GB | N/A | Q4_K_M |

### Next steps

1. **Test on larger GPU** (RTX 3090 Ti 24GB or better) - current blocker is VRAM
2. **Verify TurboQuant integration** once model loads successfully
3. **Add DSpark draft model support** for speculative decoding
4. **Benchmark**: MTP vs DSpark at concurrency [2,4,8,12,14,16,18,20,24]

### Architecture notes

```
vLLM GGUF Plugin
├── GGUFConfigParser → reads GGUF metadata directly
├── TurboQuant+ (optional)
│   ├── KV: WHT rotation + Lloyd-Max (K=4, V=4)
│   └── Weight: kurtosis-aware 3-bit MoE compression
├── MTP speculative decoding (baked nextn head)
└── DSpark (future: separate draft model GGUF)
```

## TurboQuant Usage

```bash
# Install TurboQuant+
pip install turboquant-plus-vllm

# Enable with KV + weight compression
VLLM_GGUF_TURBOQUANT=1 vllm serve <model.gguf> \
    --kv-cache-dtype turboquant_4bit_nc \
    --attention-backend CUSTOM

# KV-only (aggressive compression)
VLLM_GGUF_TURBOQUANT=1 vllm serve <model.gguf> \
    --kv-cache-dtype turboquant_4bit_nc \
    --attention-backend CUSTOM

# With MoE expert pruning (REAP-style 50% pruning)
VLLM_GGUF_TURBOQUANT=1 \
    VLLM_GGUF_TURBOQUANT_PRUNE_EXPERTS=0.5 \
    vllm serve <model.gguf> \
    --kv-cache-dtype turboquant_4bit_nc
```

## DSpark Usage

```bash
# DSpark requires separate draft model GGUF
vllm serve <target.gguf> \
    --speculative-config '{
        "method": "dspark",
        "model": "/path/to/dspark-qwen3-8b-block7-Q4_K_M.gguf",
        "num_speculative_tokens": 7
    }'
```
