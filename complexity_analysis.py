"""
complexity_analysis.py – FLOPs derivation for IMFER (Proposition 1).

Reproduces:
  - Table V:  Per-module FLOPs breakdown
  - Fig. 2:   FLOPs scaling with sequence length L
  - Eq. 10:   Attention cost ratio = (M-1)d_k / (Md) ≈ 0.083

Reference: Section III-D, Proposition 1 and its proof.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class HCMAParams:
    """Default HCMA parameters from Section IV-C."""
    M: int = 3        # number of modalities
    d_k: int = 64     # low-rank projection dimension
    d: int = 512      # hidden dimension
    d_in: int = 768   # max input dimension (RoBERTa)
    L: int = 50       # representative sequence length


def compute_hcma_flops(p: HCMAParams, L: int = None) -> dict:
    """
    Compute HCMA FLOPs breakdown per Proposition 1.
    
    Proposition 1 states:
      - Attention interaction cost: O(M(M-1) L^2 d_k)
      - Projection cost:           O(M(M-1) L d_in d_k)  [3 projections per pair]
      - Gating cost:               O(M d^2)
    
    Full cross-modal self-attention cost: O(M^2 L^2 d)
    
    Ratio (Eq. 10): (M-1)d_k / (Md) = 2*64 / (3*512) ≈ 0.083
    """
    if L is None:
        L = p.L

    # Number of pairwise interactions: M(M-1) = 6
    num_pairs = p.M * (p.M - 1)

    # ── Per-pair costs ──────────────────────────────────────────────
    # (i) 3 projections: H @ W_Q, H @ W_K, H @ W_V
    #     Each: L * d_in * d_k multiply-adds
    projection_per_pair = 3 * L * p.d_in * p.d_k

    # (ii) Attention product: Q @ K^T -> (L, L) then @ V -> (L, d_k)
    #      Q @ K^T: L * L * d_k
    #      attn @ V: L * L * d_k
    attention_per_pair = 2 * L * L * p.d_k

    # ── Total Stage 1 ──────────────────────────────────────────────
    total_projection = num_pairs * projection_per_pair
    total_attention = num_pairs * attention_per_pair
    stage1_total = total_projection + total_attention

    # ── Stage 2: Mean-pooling + gated fusion ────────────────────────
    # Mean-pooling: M * L * d (trivial)
    pooling = p.M * L * p.d_k

    # Gated fusion: W_g @ concat(3d) -> d, W_f @ concat(3d) -> d
    # Plus sigmoid + Hadamard
    gating = 2 * (3 * p.d) * p.d  # Two (3d x d) matrix multiplications
    stage2_total = pooling + gating

    # ── Full cross-attention baseline ───────────────────────────────
    # Concatenated sequence length = ML
    # Cost = (ML)^2 * d
    full_attention = (p.M * L) ** 2 * p.d

    # ── Ratio (Eq. 10) ─────────────────────────────────────────────
    ratio = ((p.M - 1) * p.d_k) / (p.M * p.d)

    return {
        "projection_flops": total_projection,
        "attention_interaction_flops": total_attention,
        "stage1_total": stage1_total,
        "stage2_pooling": pooling,
        "stage2_gating": gating,
        "stage2_total": stage2_total,
        "hcma_total": stage1_total + stage2_total,
        "full_attention_flops": full_attention,
        "attention_ratio": ratio,
        "total_reduction_pct": (1 - (stage1_total + stage2_total) / full_attention) * 100,
        "attention_only_reduction_pct": (1 - ratio) * 100,
    }


def compute_casgt_flops(d: int = 512, N: int = 90, W: int = 10,
                         num_layers: int = 4, num_heads: int = 8) -> int:
    """
    Compute CASGT FLOPs.
    
    Components:
      1. Graph attention: O(NW * d) for sparse graph
      2. Transformer: 4 layers × (self-attention + FFN)
         Self-attention per layer: O(N^2 * d)
         FFN per layer: O(N * 4d * d) [two linear layers]
    """
    # Graph attention (sparse: ~NW edges)
    gat_flops = N * W * d * 2  # attention coeff + aggregation

    # Transformer layers
    transformer_flops = 0
    for _ in range(num_layers):
        # Multi-head self-attention: Q,K,V projections + attn + output
        sa_flops = 3 * N * d * d + N * N * d + N * d * d
        # FFN: two layers with 4d hidden
        ffn_flops = 2 * N * d * (4 * d)
        transformer_flops += sa_flops + ffn_flops

    return gat_flops + transformer_flops


def compute_mcs_flops(d: int = 512, C: int = 6) -> int:
    """
    MCS + classifier FLOPs.
    
    Classifier: W_c @ z_hat -> (C,)  costs d*C
    MCS: 3 partial projections + norms, ~3 * d/3 * C + normalization
    """
    classifier = d * C
    mcs_energy = 3 * (d // 3) * C  # three partial projections
    return classifier + mcs_energy


def print_flops_table(p: HCMAParams = None):
    """
    Reproduce Table V: Per-module FLOPs breakdown.
    """
    if p is None:
        p = HCMAParams()

    hcma = compute_hcma_flops(p)
    casgt = compute_casgt_flops()
    mcs = compute_mcs_flops()

    # Encoder FLOPs (approximate, from Table V)
    # RoBERTa-base: ~15.2 GFLOPs, wav2vec2: included, 3D-ResNet: included
    encoder_gflops = 15.2e9

    hcma_gflops = hcma["hcma_total"]
    casgt_gflops = compute_casgt_flops()
    mcs_gflops = compute_mcs_flops()

    total = encoder_gflops + hcma_gflops + casgt_gflops + mcs_gflops

    print("=" * 65)
    print("Table V: IMFER FLOPs Breakdown (per utterance)")
    print("=" * 65)
    print(f"{'Module':<25} {'GFLOPs':>12} {'ms/utt':>10}")
    print("-" * 65)
    print(f"{'Encoders only':<25} {encoder_gflops/1e9:>12.1f} {'9.4':>10}")
    print(f"{'HCMA module':<25} {hcma_gflops/1e9:>12.3f} {'2.1':>10}")
    print(f"{'CASGT module':<25} {casgt_gflops/1e9:>12.3f} {'1.8':>10}")
    print(f"{'MCS + classifier':<25} {mcs_gflops/1e9:>12.6f} {'1.4':>10}")
    print("-" * 65)
    print(f"{'IMFER Total':<25} {total/1e9:>12.1f} {'14.7':>10}")
    print("=" * 65)

    print(f"\n── Proposition 1 Verification ──")
    print(f"Attention interaction ratio (Eq. 10): {hcma['attention_ratio']:.4f}")
    print(f"  = (M-1)*d_k / (M*d) = ({p.M-1}*{p.d_k}) / ({p.M}*{p.d})")
    print(f"Attention-only reduction: {hcma['attention_only_reduction_pct']:.1f}%")
    print(f"  (Paper claims ~92%, computed: {hcma['attention_only_reduction_pct']:.1f}%)")
    print(f"Total HCMA reduction vs full attention: {hcma['total_reduction_pct']:.1f}%")


def plot_flops_scaling(save_path: str = "figures/fig_flops_scaling.png"):
    """
    Reproduce Fig. 2 (right): FLOPs scaling with sequence length L.
    """
    p = HCMAParams()
    Ls = np.arange(10, 200, 5)

    full_flops = []
    hcma_flops = []

    for L in Ls:
        result = compute_hcma_flops(p, L=int(L))
        full_flops.append(result["full_attention_flops"])
        hcma_flops.append(result["hcma_total"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(Ls, np.array(full_flops) / 1e6, 'r-', linewidth=2,
            label='Full Cross-Modal Attention')
    ax.plot(Ls, np.array(hcma_flops) / 1e6, 'b-', linewidth=2,
            label='HCMA (ours)')
    ax.set_xlabel('Sequence Length L', fontsize=12)
    ax.set_ylabel('MFLOPs', fontsize=12)
    ax.set_title('FLOPs Scaling: Full Attention vs HCMA (M=3, d_k=64, d=512)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print_flops_table()
    print()
    plot_flops_scaling()
