# SPDX-License-Identifier: Apache-2.0
"""TurboQuant+ KV cache and weight compression for vLLM GGUF models.

TurboQuant+ (ICLR 2026) provides:
- KV cache compression: 3-6x smaller KV cache via WHT rotation + Lloyd-Max quantization
- Weight compression: 3-4.6x reduction with kurtosis-aware bit allocation for MoE

References:
- https://github.com/varjoranta/turboquant-vllm
- https://vllm.ai/blog/2026-05-11-turboquant
"""

from __future__ import annotations


def _turboquant_available() -> bool:
    try:
        import turboquant_vllm  # noqa: F401
        return True
    except ImportError:
        return False


def install_turboquant_kv(
    k_bits: int = 4,
    v_bits: int = 4,
    norm_correction: bool = True,
    boundary_layers: int = 5,
    sink_tokens: int = 4,
) -> bool:
    """Patch vLLM attention backends for TurboQuant+ KV cache compression.

    Args:
        k_bits: Bits for key compression (default 4, 16 centroids).
        v_bits: Bits for value compression (default 4).
        norm_correction: Correct reconstruction magnitude error (default True).
        boundary_layers: First and last N layers get K=8-bit precision (default 5).
            Boundary layers carry more signal through the residual stream.
        sink_tokens: First N positions per layer stored at FP16 (default 4).
            Attention sinks get universal attention and need full precision.

    Returns:
        True if TurboQuant was successfully installed, False if not available.
    """
    if not _turboquant_available():
        return False

    try:
        from turboquant_vllm import patch_vllm_attention
    except ImportError:
        return False

    try:
        patch_vllm_attention(
            k_bits=k_bits,
            v_bits=v_bits,
            norm_correction=norm_correction,
            boundary_layers=boundary_layers,
            sink_tokens=sink_tokens,
        )
        return True
    except Exception:
        return False


def install_turboquant_weights(
    bits: int = 3,
    group_size: int = 128,
    kurtosis_aware: bool = True,
    prune_experts: float = 0.0,
    routed_expert_bits: int | None = None,
) -> bool:
    """Patch vLLM to apply TurboQuant weight compression at model load time.

    Args:
        bits: Default quantization bits (2-8). Default 3 for best size/quality.
        group_size: Elements per quantization group. Default 128 (matches head_dim).
        kurtosis_aware: Auto-select bits per tensor based on kurtosis.
            Heavy-tailed tensors (shared MoE experts) get more bits.
            Near-Gaussian tensors (routed experts) get fewer bits.
        prune_experts: Fraction of routed MoE experts to prune (0.0-1.0).
            Uses router weight norms to rank experts by importance.
            REAP (ICLR 2026): 50% pruning retains 97.6% quality on MoE.
        routed_expert_bits: Override bit width for routed expert weights.

    Returns:
        True if weight quantization was successfully enabled, False if not available.
    """
    if not _turboquant_available():
        return False

    try:
        from turboquant_vllm import enable_weight_quantization
    except ImportError:
        return False

    try:
        enable_weight_quantization(
            bits=bits,
            group_size=group_size,
            kurtosis_aware=kurtosis_aware,
            prune_experts=prune_experts,
            routed_expert_bits=routed_expert_bits,
        )
        return True
    except Exception:
        return False


def install_turboquant(
    kv_k_bits: int = 4,
    kv_v_bits: int = 4,
    weight_bits: int = 3,
    kurtosis_aware: bool = True,
    prune_experts: float = 0.0,
    routed_expert_bits: int | None = None,
    kv_norm_correction: bool = True,
    kv_boundary_layers: int = 5,
) -> dict[str, bool]:
    """Install both TurboQuant KV cache and weight compression.

    Args:
        kv_k_bits: Bits for key compression (default 4).
        kv_v_bits: Bits for value compression (default 4).
        weight_bits: Default quantization bits for weights (default 3).
        kurtosis_aware: Auto-select bits per tensor based on kurtosis for weights.
        prune_experts: Fraction of MoE experts to prune (0.0-1.0).
        routed_expert_bits: Override bit width for routed expert weights.
        kv_norm_correction: Correct reconstruction magnitude error for KV.
        kv_boundary_layers: First/last N layers get K=8-bit for KV.

    Returns:
        dict with 'kv' and 'weights' boolean status.
    """
    kv_ok = install_turboquant_kv(
        k_bits=kv_k_bits,
        v_bits=kv_v_bits,
        norm_correction=kv_norm_correction,
        boundary_layers=kv_boundary_layers,
    )

    weight_ok = install_turboquant_weights(
        bits=weight_bits,
        kurtosis_aware=kurtosis_aware,
        prune_experts=prune_experts,
        routed_expert_bits=routed_expert_bits,
    )

    return {"kv": kv_ok, "weights": weight_ok}


def is_turboquant_installed() -> bool:
    """Check if TurboQuant KV patching is active."""
    if not _turboquant_available():
        return False
    try:
        from turboquant_vllm.vllm_patch import _cache
        return len(_cache) > 0
    except Exception:
        return False
