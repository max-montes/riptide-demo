# Riptide Binary Convolution Demo

A from-scratch implementation of binary convolution — the core technique behind [Riptide: Fast End-to-End Binarized Neural Networks](https://proceedings.mlsys.org/paper_files/paper/2020/hash/521437c574a2bb7fcc20b222700b4181-Abstract.html) (MLSys 2020).

## What Are Binarized Neural Networks?

Standard neural networks use 32-bit floating-point weights and activations. BNNs constrain both to just **+1 or -1**, replacing expensive multiply-accumulate operations with **XNOR + popcount** — bitwise ops that are orders of magnitude faster.

```
Standard:  output = Σ (weight_i × activation_i)   # Float multiply-accumulate
Binary:    output = 2 * popcount(XNOR(w, a)) - n  # Bitwise ops
```

## What's In This Repo

`binary_conv.py` implements three versions of 2D convolution:

| Method | Description | Operations per output pixel |
|--------|-------------|---------------------------|
| **Standard** | Float32 multiply-accumulate | 9 multiplies + 8 adds (3×3 kernel) |
| **Binary (naive)** | Binarize then count matches | 9 comparisons + counting |
| **Binary (XNOR)** | Bit-pack → XNOR → popcount | 1 XNOR + 1 popcount |

The demo walks through each step:
1. **Binarization** — converting float values to +1/-1
2. **Bit packing** — compressing binary values into integers
3. **XNOR + popcount** — replacing multiplication with bitwise ops
4. **Verification** — proving XNOR matches the naive implementation
5. **Benchmarking** — comparing performance (with caveats)

## Running

```bash
pip install numpy
python binary_conv.py
```

## Example Output

```
XNOR + POPCOUNT WALKTHROUGH (first patch)

Packed patch:  101010010 (338)    ← 9 binary values in one integer
Packed kernel: 110110110 (438)
XNOR result:   100011011 (283)    ← 1 where bits match
Popcount:      5                   ← 5 matches out of 9
Dot product:   2 * 5 - 9 = 1
```

## Why Python Won't Show Real Speedups

This implementation is educational. Real BNN speedups (4-12×) require:
- **Packing 64 values** into a single `uint64` (we only pack 9)
- **Hardware POPCNT** instruction (single cycle on modern CPUs)
- **SIMD/NEON vectorization** for parallel bitwise ops
- **TVM-optimized memory layouts** and kernel scheduling
- **Fused Glue operator** — Riptide's key innovation that eliminates floating-point ops between binary layers

## The Riptide Paper

**"Riptide: Fast End-to-End Binarized Neural Networks"** — Fromm, Cowan, Philipose, Ceze, Patel (MLSys 2020)

Key contributions:
- **Fused Glue**: Merges BatchNorm + binarization into a single threshold comparison
- **Bitpack Fusion**: Keeps values packed across layers, minimizing pack/unpack overhead
- **TVM kernels**: Auto-generated optimized binary convolution kernels for ARM/x86
- **First measured end-to-end speedups**: 4-12× on Raspberry Pi 3B

Official implementation: [github.com/jwfromm/Riptide](https://github.com/jwfromm/Riptide)

## References

- [Riptide paper (MLSys 2020)](https://proceedings.mlsys.org/paper_files/paper/2020/hash/521437c574a2bb7fcc20b222700b4181-Abstract.html)
- [Official Riptide repo](https://github.com/jwfromm/Riptide)
- [XNOR-Net (Rastegari et al., 2016)](https://arxiv.org/abs/1603.05279) — foundational BNN work
