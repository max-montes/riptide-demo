"""
Binary Convolution Demo
=======================
Demonstrates the core idea behind Binarized Neural Networks (BNNs):
- Weights and activations constrained to +1/-1
- Multiply-accumulate replaced with XNOR + popcount
- Massive speedup potential on edge devices

This implements the concepts from:
"Riptide: Fast End-to-End Binarized Neural Networks" (MLSys 2020)
"""

import numpy as np
import time


# =============================================================================
# STEP 1: Binarization
# =============================================================================

def binarize(x: np.ndarray) -> np.ndarray:
    """
    Binarize values to +1 or -1 using the sign function.
    
    In BNNs:
      - Positive values -> +1
      - Negative values -> -1
      - Zero -> +1 (convention)
    """
    return np.where(x >= 0, 1, -1).astype(np.int8)


def binarize_to_bits(x: np.ndarray) -> np.ndarray:
    """
    Binarize to 0/1 for bit packing.
    
      - Positive values -> 1
      - Negative values -> 0
    """
    return (x >= 0).astype(np.uint8)


# =============================================================================
# STEP 2: Standard Convolution (Baseline)
# =============================================================================

def conv2d_standard(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Standard 2D convolution with float multiply-accumulate.
    
    Args:
        input: Shape (H, W) - single channel input
        kernel: Shape (kH, kW) - convolution filter
    
    Returns:
        output: Shape (H-kH+1, W-kW+1)
    """
    H, W = input.shape
    kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    
    output = np.zeros((out_H, out_W), dtype=np.float32)
    
    for i in range(out_H):
        for j in range(out_W):
            # Extract patch and compute dot product
            patch = input[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * kernel)  # Multiply-accumulate
    
    return output


# =============================================================================
# STEP 3: Binary Convolution (Naive)
# =============================================================================

def conv2d_binary_naive(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Binary convolution using +1/-1 values.
    
    When both values are +1 or -1:
      (+1) * (+1) = +1
      (+1) * (-1) = -1
      (-1) * (-1) = +1
    
    This is equivalent to: 2 * (a == b) - 1
    Or simply: count matches - count mismatches
    """
    # Binarize inputs
    input_bin = binarize(input)
    kernel_bin = binarize(kernel)
    
    H, W = input_bin.shape
    kH, kW = kernel_bin.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    
    output = np.zeros((out_H, out_W), dtype=np.float32)
    
    for i in range(out_H):
        for j in range(out_W):
            patch = input_bin[i:i+kH, j:j+kW]
            # Count where signs match vs don't match
            matches = np.sum(patch == kernel_bin)
            mismatches = kH * kW - matches
            output[i, j] = matches - mismatches
    
    return output


# =============================================================================
# STEP 4: Binary Convolution with XNOR + Popcount
# =============================================================================

def pack_bits(binary_array: np.ndarray) -> int:
    """
    Pack a binary array (0s and 1s) into a single integer.
    
    Example: [1, 0, 1, 1] -> 0b1011 = 11
    """
    result = 0
    for bit in binary_array.flatten():
        result = (result << 1) | int(bit)
    return result


def popcount(x: int) -> int:
    """
    Count the number of 1-bits in an integer.
    
    This maps directly to a single CPU instruction (POPCNT).
    """
    return bin(x).count('1')


def xnor(a: int, b: int, num_bits: int) -> int:
    """
    XNOR operation: returns 1 where bits match, 0 where they differ.
    
    XNOR truth table:
      a | b | XNOR
      0 | 0 |  1    (both negative -> product is +1)
      0 | 1 |  0    (different signs -> product is -1)
      1 | 0 |  0    (different signs -> product is -1)
      1 | 1 |  1    (both positive -> product is +1)
    """
    # XOR gives 1 where different, so we invert and mask to num_bits
    mask = (1 << num_bits) - 1
    return (~(a ^ b)) & mask


def conv2d_binary_xnor(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Binary convolution using XNOR + popcount.
    
    This is the key insight of BNNs:
      1. Pack binary values into integers
      2. XNOR gives 1 where signs match (product would be +1)
      3. Popcount counts matches in one CPU instruction
      4. Result = 2 * matches - total_bits
    
    On real hardware, this replaces N multiply-accumulates with:
      - 1 XNOR instruction
      - 1 POPCOUNT instruction
    """
    # Binarize to 0/1 for bit packing
    input_bits = binarize_to_bits(input)
    kernel_bits = binarize_to_bits(kernel)
    
    H, W = input_bits.shape
    kH, kW = kernel_bits.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    num_bits = kH * kW
    
    # Pack kernel once (weights are fixed at inference)
    kernel_packed = pack_bits(kernel_bits)
    
    output = np.zeros((out_H, out_W), dtype=np.float32)
    
    for i in range(out_H):
        for j in range(out_W):
            # Pack input patch
            patch = input_bits[i:i+kH, j:j+kW]
            patch_packed = pack_bits(patch)
            
            # XNOR + popcount
            xnor_result = xnor(patch_packed, kernel_packed, num_bits)
            matches = popcount(xnor_result)
            
            # Convert to dot product: matches contribute +1, mismatches -1
            output[i, j] = 2 * matches - num_bits
    
    return output


# =============================================================================
# STEP 5: Demonstration
# =============================================================================

def demo():
    print("=" * 60)
    print("BINARY CONVOLUTION DEMO")
    print("=" * 60)
    
    # Create sample input and kernel
    np.random.seed(42)
    input_size = 8
    kernel_size = 3
    
    # Random float input (simulating activations from previous layer)
    input_data = np.random.randn(input_size, input_size).astype(np.float32)
    
    # Random float kernel (simulating learned weights)
    kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
    
    print(f"\nInput shape: {input_data.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: ({input_size - kernel_size + 1}, {input_size - kernel_size + 1})")
    
    # Show binarization
    print("\n" + "-" * 60)
    print("BINARIZATION EXAMPLE")
    print("-" * 60)
    print(f"\nOriginal kernel:\n{kernel.round(2)}")
    print(f"\nBinarized kernel (+1/-1):\n{binarize(kernel)}")
    print(f"\nBinarized kernel (0/1 for packing):\n{binarize_to_bits(kernel)}")
    
    # Run all three methods
    print("\n" + "-" * 60)
    print("CONVOLUTION RESULTS")
    print("-" * 60)
    
    result_standard = conv2d_standard(input_data, kernel)
    result_binary_naive = conv2d_binary_naive(input_data, kernel)
    result_binary_xnor = conv2d_binary_xnor(input_data, kernel)
    
    print(f"\nStandard conv (float):\n{result_standard.round(2)}")
    print(f"\nBinary conv (naive):\n{result_binary_naive.astype(int)}")
    print(f"\nBinary conv (XNOR):\n{result_binary_xnor.astype(int)}")
    
    # Verify XNOR matches naive
    assert np.allclose(result_binary_naive, result_binary_xnor), "XNOR should match naive!"
    print("\n✓ XNOR result matches naive implementation")
    
    # Show the XNOR + popcount in detail for one patch
    print("\n" + "-" * 60)
    print("XNOR + POPCOUNT WALKTHROUGH (first patch)")
    print("-" * 60)
    
    patch = input_data[0:kernel_size, 0:kernel_size]
    patch_bits = binarize_to_bits(patch)
    kernel_bits = binarize_to_bits(kernel)
    
    print(f"\nInput patch (float):\n{patch.round(2)}")
    print(f"\nInput patch (bits):\n{patch_bits}")
    print(f"\nKernel (bits):\n{kernel_bits}")
    
    patch_packed = pack_bits(patch_bits)
    kernel_packed = pack_bits(kernel_bits)
    num_bits = kernel_size * kernel_size
    
    print(f"\nPacked patch:  {patch_packed:0{num_bits}b} ({patch_packed})")
    print(f"Packed kernel: {kernel_packed:0{num_bits}b} ({kernel_packed})")
    
    xnor_result = xnor(patch_packed, kernel_packed, num_bits)
    matches = popcount(xnor_result)
    
    print(f"XNOR result:   {xnor_result:0{num_bits}b} ({xnor_result})")
    print(f"Popcount (matches): {matches}")
    print(f"Dot product: 2 * {matches} - {num_bits} = {2 * matches - num_bits}")


def benchmark():
    """Compare performance of standard vs binary convolution."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Larger input for meaningful timing
    input_size = 56  # Typical feature map size
    kernel_size = 3
    num_iterations = 10
    
    input_data = np.random.randn(input_size, input_size).astype(np.float32)
    kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
    
    print(f"\nInput: {input_size}x{input_size}, Kernel: {kernel_size}x{kernel_size}")
    print(f"Iterations: {num_iterations}")
    
    # Benchmark standard conv
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = conv2d_standard(input_data, kernel)
    time_standard = (time.perf_counter() - start) / num_iterations * 1000
    
    # Benchmark binary conv (XNOR)
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = conv2d_binary_xnor(input_data, kernel)
    time_binary = (time.perf_counter() - start) / num_iterations * 1000
    
    print(f"\nStandard conv: {time_standard:.2f} ms")
    print(f"Binary conv:   {time_binary:.2f} ms")
    print(f"Ratio:         {time_standard/time_binary:.2f}x")
    
    print("\n⚠️  Note: This Python implementation won't show speedups!")
    print("   Real gains come from:")
    print("   - Packing 64 values into one uint64")
    print("   - Hardware XNOR + POPCNT instructions")
    print("   - Optimized memory layouts (TVM)")
    print("   - SIMD/NEON vectorization")


if __name__ == "__main__":
    demo()
    benchmark()
