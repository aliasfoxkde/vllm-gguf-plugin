# vLLM GGUF Plugin — Qwen3.5/3.6 hybrid fork

Fork of [vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)
(the out-of-tree GGUF quantization plugin for vLLM ≥0.25, after in-tree GGUF
deprecation, [vllm#39583](https://github.com/vllm-project/vllm/issues/39583))
extended to serve **Qwen3.5 / Qwen3.6 hybrid (GatedDeltaNet + full-attention)
GGUFs — dense and MoE — at ik_llama.cpp-class speeds, including the baked MTP
(nextn) head as vLLM speculative decoding.**

**Maintained at [aliasfoxkde/vllm-gguf-plugin](https://github.com/aliasfoxkde/vllm-gguf-plugin).**

## Validated models

### Qwen3.6 hybrid (localweights)
- `Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn` — dense, 64 layers = 48 GDN + 16 full-attn
  [[model card](https://huggingface.co/localweights/Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF)]
  (original Qwen3.6 hybrid from localweights)
- `Qwen3.6-35B-A3B-MTP-IMAT-IQ4_XS-Q8nextn` — MoE, 256 experts
  [[model card](https://huggingface.co/localweights/Qwen3.6-35B-A3B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF)]
  (original Qwen3.6 MoE hybrid from localweights)

### Qwen3.6 REAP (JZC973)
- **Qwen3.6-35B-A3B-REAP** — MoE with REAP (Rotary Embedding Alignment with
  Alibi Positional) expert pruning, 192 experts (ATBender) or 180 experts
  (RangerX), UD quantizations.
  [[collection card](https://huggingface.co/JZC973/Qwen3.6-35B-REAP-MTP-UD-GGUF-Collection)]

  | Quantization | Size | Experts |
  |---|---|---|
  | UD-IQ3_XXS | ~10.5 GB | 192 (ATBender) |
  | UD-IQ3_S | ~11.1 GB | 192 (ATBender) |
  | UD-Q3_K_M | ~12 GB | 192 (ATBender) |
  | UD-Q3_K_XL | ~13.4 GB | 192 (ATBender) |
  | UD-IQ3_XXS (RangerX) | ~10.5 GB | 180 (RangerX) |
  | UD-IQ3_S (RangerX) | ~11.1 GB | 180 (RangerX) |
  | UD-Q3_K_M (RangerX) | ~12 GB | 180 (RangerX) |
  | UD-Q3_K_XL (RangerX) | ~13.4 GB | 180 (RangerX) |

## What this fork adds over upstream

### Qwen3.5/3.6 hybrid support (dense + MoE)
- GGUF→HF config synthesis for `qwen35` / `qwen35moe` archs (`qwen35_config.py`).
  Handles the `block_count - nextn_predict_layers` correction for MTP layer
  stripping and maps `ssm.*` → `linear_*` heads.
- Full tensor-name mapping incl. undoing llama.cpp's GDN weight transforms at
  load, merged-layer shard handling, MoE expert + `shared_expert_gate` mapping.
- Hybrid model registration; 262k-context serving on 24GB
  (`--kv-cache-dtype fp8 --enable-prefix-caching --mamba-cache-mode align`).

### MTP (nextn) speculative decoding from the baked head
- The GGUFs ship the MTP head as `blk.<N>.nextn.*` tensors; this fork maps them
  into vLLM's `Qwen3_5MTP` / `Qwen3_5MoeMTP` drafters (`mtp_enable.py`) —
  dense AND MoE (router, fused experts, shared expert + gate).
- `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` works
  directly off the single GGUF file. Inert unless the flag is passed.
- Embed/lm_head shared with the target model (quantized-weight-safe).
- Supports `num_speculative_tokens_per_batch_size` schedule (spec-on at batch 1,
  off at batch ≥2 — recommended for dense; see measured results).

### IQ4_XS / small-batch CUDA kernel work (`csrc/gguf/`)
- **Register-only PRMT nibble lookup** for iq4_xs/iq4_nl vec-dot (from
  ik_llama.cpp): 8 byte-indexed memory loads/word → 2 `__byte_perm`; kernel
  743→884 GB/s.
- **Single 64-bit block-header load** (d + scales_h + scales_l) in
  `vec_dot_iq4_xs_q8_1`.
- **Multi-column dst kernels for small batch (nvecs 2–8)** — upstream's
  `blockIdx.y = column` layout re-reads the entire weight matrix per dst
  column. One thread now accumulates all dst columns per weight fetch; fused
  2-col iq4_xs vec-dot for the MTP verify shape. Applied to iq4_xs, iq4_nl,
  q8_0, q4_K, q5_K, q6_K, q3_k. Bitwise-identical outputs.
- **2-warp thread blocks** for the small-batch kernels — 1-warp blocks cap
  Ampere at 16 resident warps/SM and starve ffn_down of latency hiding.
- Contiguity guards on the custom-op call sites — torch.compile graph regions
  pass non-contiguous views; raw `data_ptr()` kernels read out of bounds
  (intermittent CUDA illegal memory access under CUDA graphs).

## Measured results (RTX 3090 Ti 24GB @350W, single stream, temp 0)

| model | upstream plugin | this fork (base) | this fork (MTP k=1) | ik_llama.cpp (MTP) |
|---|---|---|---|---|
| Qwen3.6-27B IQ4_XS | ~29 tok/s (enforce-eager) | 44.0 | **64.4** | 79 |
| Qwen3.6-35B-A3B IQ4_XS | ~41 tok/s | 179.6 | **203.6** | 200 |

- Quality gates on every kernel/MTP change: bitwise-identical kernel outputs,
  10-prompt graded eval vs the AWQ reference, long-context needle recall.
- MTP acceptance: ~82% dense / ~69% MoE at 1 spec token. k=1 beats k=2/k=3 on
  both models on this stack.
- Batch caveat: MTP verify never amortizes on MoE (each spec token routes its
  own experts) — for multi-stream MoE serving run without MTP; for dense use
  the per-batch-size schedule.

## Usage

### Installation

```bash
pip install -e . --torch-backend=auto   # builds _C_gguf ext (set TORCH_CUDA_ARCH_LIST)
```

To reuse a pre-built extension (when nvcc version mismatches torch's CUDA):
```bash
VLLM_GGUF_PLUGIN_SKIP_EXT=1 pip install -e .
```

### Qwen3.6 REAP (JZC973)

```bash
vllm serve JZC973/Qwen3.6-35B-REAP-MTP-UD-GGUF-Collection:Qwen3.6-35B-A3B-UD-Q3_K_XL-REAP \
  --served-model-name qwen3.6-35b-reap \
  --kv-cache-dtype fp8 --enable-prefix-caching --mamba-cache-mode align \
  --gpu-memory-utilization 0.93 --max-model-len 131072 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2]}'
```

Or with MTP speculative decoding:
```bash
vllm serve JZC973/Qwen3.6-35B-REAP-MTP-UD-GGUF-Collection:Qwen3.6-35B-A3B-UD-Q3_K_XL-REAP \
  --served-model-name qwen3.6-35b-reap \
  --kv-cache-dtype fp8 --enable-prefix-caching --mamba-cache-mode align \
  --gpu-memory-utilization 0.93 --max-model-len 131072 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### Qwen3.6 hybrid (localweights) with MTP

```bash
vllm serve /path/Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn.gguf \
  --served-model-name qwen3.6-27b-iq4xs \
  --hf-config-path /path/to/config-dir \
  --kv-cache-dtype fp8 --enable-prefix-caching --mamba-cache-mode align \
  --gpu-memory-utilization 0.93 --max-model-len 131072 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### General GGUF (upstream-compatible)

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```

## Notes

- Do NOT pass `--enforce-eager` — CUDA graphs are ~1.5–3.9× decode speedup on
  these hybrids.
- Cap `cudagraph_capture_sizes` at your real concurrency; a spec-decode step is
  `batch × (1+k)` tokens and uncaptured spec steps fall off a cliff.
- For MoE models, MTP verify does not amortize (each spec token routes its own
  experts) — run without MTP for multi-stream MoE serving.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit run --all-files

# Run tests (skip slow GPU tests)
pytest -m "not slow"
pytest tests/test_kernels.py   # single file
```

## Repository

Forked from [vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin).
Maintained at [aliasfoxkde/vllm-gguf-plugin](https://github.com/aliasfoxkde/vllm-gguf-plugin)
(PRs/issues welcome).
