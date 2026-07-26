# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM GGUF Plugin fork extended to serve **Qwen3.5 / Qwen3.6 hybrid (GDN + full-attention) GGUFs** — dense and MoE. Validated on single RTX 3090 Ti 24GB with vLLM 0.25.0.

Key features added over upstream:
- GGUF→HF config synthesis for `qwen35` / `qwen35moe` archs
- IQ4_XS / small-batch CUDA kernel optimizations
- MTP (nextn) speculative decoding from baked head
- Hybrid model registration with 262k context support

## Build & Install

```bash
pip install -e . --torch-backend=auto   # builds _C_gguf ext (set TORCH_CUDA_ARCH_LIST)
```

To skip extension rebuild (reuse existing `.so`):
```bash
VLLM_GGUF_PLUGIN_SKIP_EXT=1 pip install -e .
```

## Development

### Pre-commit hooks
```bash
pre-commit run --all-files
```

Hooks: `ruff` (lint + format), `typos`, `clang-format` (CUDA/C++), `markdownlint-cli2`, filename-space check.

### Running tests
```bash
pytest                                    # all tests
pytest tests/test_kernels.py              # single file
pytest -m "not slow"                     # skip slow GPU tests
VLLM_GGUF_PLUGIN_SKIP_EXT=1 pytest       # use pre-built .so
```

Slow tests are marked with `@pytest.mark.slow` (require GPU/large model downloads).

### Linting (manual)
```bash
ruff check .
ruff format .
clang-format --style=file vllm_gguf_plugin/csrc/gguf/*.cu
```

## Architecture

```
vllm_gguf_plugin/
├── __init__.py              # Entry: install_idempotent_op_registration() before anything else
├── plugin.py                 # register() — patches vLLM EngineArgs, model loader, config parser
├── config_parser.py          # GGUFConfigParser — parses GGUF files into vLLM configs
├── loader.py                 # GGUFModelLoader — downloads/resolves GGUF, loads into model
├── qwen35_config.py          # map_qwen35_config() + register() — monkeypatches GGUF loader
│                             #   for qwen35/qwen35moe archs (not in upstream GGUF_SUPPORTED_ARCHITECTURES)
├── mtp_enable.py             # MTP (nextn) speculative decoding from blk.<N>.nextn.* tensors
├── mtp_eagle_group.py        # eagle_group annotation patching for MTP
├── mtp_dynamic_skip.py       # Skip MTP layers at runtime when not speculating
├── sleep_wake_hybrid.py      # FP8 KV cache scale reset on wake from sleep mode
├── offload_instrument.py     # Transfer bounds validator (Xid 31 evidence trap)
├── op_guard.py               # install_idempotent_op_registration() — prevents duplicate op registration
├── ops.py                    # Python bindings for CUDA ops
├── gguf_utils.py             # GGUF file inspection utilities
├── weight_utils.py           # GGUF download, resolution, tensor iteration
├── spec_plan_cache.py        # Speculative plan caching
├── csrc/                     # CUDA/C++ extensions
│   ├── gguf/gguf_kernel.cu  # Main CUDA kernel entry point
│   ├── gguf/vecdotq.cuh      # Vector dot product kernels (iq4_xs, q8_0, q4_K, etc.)
│   ├── gguf/moe.cuh          # MoE CUDA kernels
│   ├── gguf/moe_vec.cuh      # MoE vector kernels
│   └── gguf/mmq.cuh, mmvq.cuh, dequantize.cuh
├── quantization/             # vLLM quantization integration
│   ├── config.py             # GGUFConfig — quantization config
│   ├── linear.py             # GGUFLinearMethod — fused mul mat
│   ├── fused_moe.py          # GGUFMoEMethod — MoE fusion
│   └── params.py             # GGUFWeightParameter etc.
└── weights_adapter/          # Tensor name mapping and weight loading
    ├── base.py               # BaseGGUFWeightsAdapter, GGUFLoadSpec
    └── default.py            # DefaultGGUFWeightsAdapter — main adapter with
                              #   qwen35/qwen35moe tensor mapping, GDN layout fixups
```

### Key design patterns

**Registration order**: `op_guard.install_idempotent_op_registration()` must run in `__init__.py` *before* `.quantization` imports — prevents duplicate `vllm::` custom op registration.

**GGUF loading flow**: `plugin.register()` patches `EngineArgs.create_model_config` to detect `.gguf` refs → sets `quantization=gguf`, `load_format=gguf` → `GGUFModelLoader` resolves the file → `get_weights_adapter()` returns a `BaseGGUFWeightsAdapter` (e.g. `DefaultGGUFWeightsAdapter`) → adapter maps GGUF tensor names to HF names and yields `(name, tensor)` pairs.

**Qwen3.6 hybrid**: Qwen3.6 has 48 GDN layers + 16 full-attn layers in `blk.<N>.*` where N=0-47=GDN, N=48-63=full-attn. The `DefaultGGUFWeightsAdapter` handles GDN layout transforms (`_inverse_reorder_v_heads`) and MoE expert + shared expert mapping.

**MTP (nextn)**: The GGUF stores MTP head tensors as `blk.<N>.nextn.*`. `mtp_enable.py` maps these into vLLM's `Qwen3_5MTP` / `Qwen3_5MoeMTP` drafters. Enabled via `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.

**CUDA kernels**: `gguf_kernel.cu` is the main entry point exposing `vec_dot_iq4_xs_q8_1` etc. via `torch::stable::ops`. Kernels use 2-warp thread blocks for small-batch shapes to avoid Ampere's 16-blocks/SM limit starving ffn_down.
