"""
Fused Glue Operator Demo
=========================
Implements Riptide's key innovation: fusing BatchNorm + Binarization
into a single threshold comparison, eliminating floating-point overhead
between binary convolution layers.

This is the #1 reason Riptide achieved real measured speedups where
previous BNN papers could not.

Reference:
"Riptide: Fast End-to-End Binarized Neural Networks" (MLSys 2020)
"""

import numpy as np
import time


# =============================================================================
# BACKGROUND: Why BatchNorm Exists
# =============================================================================
#
# After a convolution, activations can drift to arbitrary ranges.
# BatchNorm normalizes them to mean=0, std=1, then applies learned
# scale (gamma) and shift (beta):
#
#   BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
#
# This helps training converge and is used in virtually every modern CNN.
#
# THE PROBLEM IN BNNs:
#   Binary Conv (fast!) → BatchNorm (FLOAT!) → Sign (FLOAT!) → Binary Conv
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                         These float ops dominate runtime because
#                         the binary conv is so fast.


# =============================================================================
# STEP 1: Standard BatchNorm + Binarization (Unfused)
# =============================================================================

def batch_norm(x: np.ndarray, mean: float, var: float,
               gamma: float, beta: float, eps: float = 1e-5) -> np.ndarray:
    """
    Standard Batch Normalization.

    4 float operations per element:
      1. Subtract mean
      2. Divide by sqrt(var + eps)
      3. Multiply by gamma
      4. Add beta
    """
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def binarize_sign(x: np.ndarray) -> np.ndarray:
    """
    Standard binarization: sign function.

    1 comparison per element.
    """
    return np.where(x >= 0, 1, -1).astype(np.int8)


def unfused_bn_then_binarize(x: np.ndarray, mean: float, var: float,
                              gamma: float, beta: float) -> np.ndarray:
    """
    The naive approach: BatchNorm, then binarize separately.

    Cost per element: 4 float ops (BN) + 1 comparison (sign) = 5 ops
    """
    normalized = batch_norm(x, mean, var, gamma, beta)
    return binarize_sign(normalized)


# =============================================================================
# STEP 2: The Fused Glue Insight
# =============================================================================
#
# Key observation: We immediately throw away the BN output by binarizing it.
# We only care about the SIGN of BN(x), not its exact value.
#
# BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
#
# sign(BN(x)) = sign(gamma * (x - mean) / sqrt(var + eps) + beta)
#
# Since sqrt(var + eps) > 0, and assuming gamma > 0:
#
#   sign(BN(x)) = sign(x - mean + beta * sqrt(var + eps) / gamma)
#               = sign(x - threshold)
#
# Where: threshold = mean - beta * sqrt(var + eps) / gamma
#
# If gamma < 0, the sign flips:
#   sign(BN(x)) = -sign(x - threshold)
#
# RESULT: 4 float ops + 1 comparison → 1 comparison against a precomputed threshold!


def compute_fused_threshold(mean: float, var: float,
                            gamma: float, beta: float,
                            eps: float = 1e-5) -> tuple[float, bool]:
    """
    Precompute the threshold that replaces BatchNorm + Sign.

    This is computed ONCE at model conversion time, not per-inference.

    Returns:
        threshold: The value to compare against
        flip_sign: Whether to flip the output (when gamma < 0)
    """
    std = np.sqrt(var + eps)
    threshold = mean - (beta * std / gamma)
    flip_sign = gamma < 0
    return threshold, flip_sign


def fused_glue(x: np.ndarray, threshold: float,
               flip_sign: bool = False) -> np.ndarray:
    """
    Riptide's Fused Glue operator.

    Replaces: BatchNorm → Sign
    With:     Single threshold comparison

    Cost per element: 1 comparison. That's it.

    This is WHY Riptide achieves real speedups:
      Before: BinaryConv (fast) → BN (slow!) → Sign (slow!) → BinaryConv
      After:  BinaryConv (fast) → Fused Glue (fast!) → BinaryConv
    """
    result = np.where(x >= threshold, 1, -1).astype(np.int8)
    if flip_sign:
        result = -result
    return result


# =============================================================================
# STEP 3: Demonstration
# =============================================================================

def demo():
    print("=" * 60)
    print("FUSED GLUE OPERATOR DEMO")
    print("=" * 60)

    np.random.seed(42)

    # Simulate output from a binary conv layer (before normalization)
    conv_output = np.random.randn(4, 4).astype(np.float32) * 3 + 1

    # Learned BatchNorm parameters (these come from training)
    mean = 1.2
    var = 8.5
    gamma = 0.8
    beta = -0.3

    print(f"\nConv output (raw):\n{conv_output.round(2)}")
    print(f"\nBatchNorm params: mean={mean}, var={var}, gamma={gamma}, beta={beta}")

    # --- Unfused path ---
    print("\n" + "-" * 60)
    print("UNFUSED: BatchNorm → Sign (5 ops per element)")
    print("-" * 60)

    bn_output = batch_norm(conv_output, mean, var, gamma, beta)
    print(f"\nAfter BatchNorm (float):\n{bn_output.round(4)}")

    unfused_result = binarize_sign(bn_output)
    print(f"\nAfter Sign:\n{unfused_result}")

    # --- Fused path ---
    print("\n" + "-" * 60)
    print("FUSED GLUE: Single threshold (1 op per element)")
    print("-" * 60)

    threshold, flip_sign = compute_fused_threshold(mean, var, gamma, beta)
    print(f"\nPrecomputed threshold: {threshold:.4f}")
    print(f"Flip sign: {flip_sign}")
    print(f"(computed ONCE, reused for every inference)")

    fused_result = fused_glue(conv_output, threshold, flip_sign)
    print(f"\nAfter Fused Glue:\n{fused_result}")

    # --- Verify they match ---
    assert np.array_equal(unfused_result, fused_result), "Results should match!"
    print("\n✓ Fused result matches unfused result exactly")

    # --- Show the math ---
    print("\n" + "-" * 60)
    print("THE MATH (for one element)")
    print("-" * 60)

    x = conv_output[0, 0]
    std = np.sqrt(var + 1e-5)
    bn_val = gamma * (x - mean) / std + beta

    print(f"\n  x = {x:.4f}")
    print(f"\n  Unfused path:")
    print(f"    BN(x) = {gamma} * ({x:.4f} - {mean}) / sqrt({var}) + ({beta})")
    print(f"         = {gamma} * {x - mean:.4f} / {std:.4f} + ({beta})")
    print(f"         = {bn_val:.4f}")
    print(f"    sign({bn_val:.4f}) = {1 if bn_val >= 0 else -1}")
    print(f"\n  Fused path:")
    print(f"    threshold = {mean} - ({beta} * {std:.4f} / {gamma}) = {threshold:.4f}")
    print(f"    x >= threshold?  {x:.4f} >= {threshold:.4f}?  {'Yes → +1' if x >= threshold else 'No → -1'}")
    print(f"\n  Same result, but 1 comparison instead of 5 float ops!")


def benchmark():
    """Show the op count difference between fused and unfused."""
    print("\n" + "=" * 60)
    print("PERFORMANCE: OP COUNT COMPARISON")
    print("=" * 60)

    np.random.seed(42)
    size = 224 * 224  # Typical feature map
    x = np.random.randn(size).astype(np.float32)

    mean, var, gamma, beta = 0.5, 2.0, 1.0, 0.0
    threshold, flip = compute_fused_threshold(mean, var, gamma, beta)

    num_iterations = 1000

    # Benchmark unfused
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = unfused_bn_then_binarize(x, mean, var, gamma, beta)
    time_unfused = (time.perf_counter() - start) / num_iterations * 1000

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = fused_glue(x, threshold, flip)
    time_fused = (time.perf_counter() - start) / num_iterations * 1000

    print(f"\nFeature map size: {size:,} elements")
    print(f"Iterations: {num_iterations}")
    print(f"\nUnfused (BN + Sign): {time_unfused:.3f} ms")
    print(f"Fused Glue:          {time_fused:.3f} ms")
    print(f"Speedup:             {time_unfused/time_fused:.1f}×")

    total_elements = size
    print(f"\n  Unfused: {total_elements * 5:,} float ops  (4 BN + 1 sign)")
    print(f"  Fused:   {total_elements * 1:,} comparisons (threshold only)")
    print(f"  Ops eliminated: {total_elements * 4:,} ({80}%)")

    print("\n" + "-" * 60)
    print("IN A FULL NETWORK")
    print("-" * 60)

    num_layers = 10
    print(f"\n  Assume {num_layers} binary conv layers:")
    print(f"  Unfused: {num_layers} × (BinaryConv + BN + Sign)")
    print(f"         = {num_layers} fast ops + {num_layers * 2} slow float ops")
    print(f"  Fused:   {num_layers} × (BinaryConv + FusedGlue)")
    print(f"         = {num_layers * 2} fast ops, zero unnecessary float ops")
    print(f"\n  The float ops between layers were the bottleneck.")
    print(f"  Fused Glue removes them → binary conv speed is no longer masked.")


if __name__ == "__main__":
    demo()
    benchmark()
