"""
models.py – IMFER model architecture.

Implements:
  - Hierarchical Cross-Modal Attention (HCMA)        [Section III-D, Eq. 1-4]
  - Context-Aware Speaker Graph Transformer (CASGT)   [Section III-C]
  - Modality Contribution Score (MCS) layer           [Section III-E, Eq. 5-6]
  - Full IMFER framework                              [Section III-A]

All equation numbers reference the paper.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# 1. HCMA: Hierarchical Cross-Modal Attention
#    Paper Section III-D, Equations 1-4
# ─────────────────────────────────────────────────────────────────────

class TokenLevelCrossModalAttention(nn.Module):
    """
    Stage 1 of HCMA: Token-level cross-modal attention with low-rank
    projection (Eq. 1-2).
    
    For modality pair (m, n):
        Q^m = H^m @ W_Q  ∈ R^{L_m × d_k}   (Eq. 2)
        K^n = H^n @ W_K  ∈ R^{L_n × d_k}   (Eq. 2)
        V^n = H^n @ W_V  ∈ R^{L_n × d_k}   (Eq. 2)
        A^{mn} = softmax(Q^m @ K^n.T / sqrt(d_k)) @ V^n   (Eq. 1)
    
    This is the KEY efficiency innovation: by projecting into d_k << d,
    the attention interaction cost drops from O(L^2 d) to O(L^2 d_k).
    """

    def __init__(self, d_in: int, d_k: int, dropout: float = 0.3):
        """
        Args:
            d_in:  input dimension of the source modality (d_text, d_audio, or d_visual)
            d_k:   low-rank projection dimension (default 64, Section IV-C)
            dropout: attention dropout rate
        """
        super().__init__()
        self.d_k = d_k
        # Shared projection matrices W_Q, W_K, W_V ∈ R^{d_in × d_k} (Eq. 2)
        self.W_Q = nn.Linear(d_in, d_k, bias=False)
        self.W_K = nn.Linear(d_in, d_k, bias=False)
        self.W_V = nn.Linear(d_in, d_k, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_m: torch.Tensor,   # query modality: (B, L_m, d_in)
        H_n: torch.Tensor,   # key/value modality: (B, L_n, d_in)
    ) -> torch.Tensor:
        """
        Compute cross-modal attention A^{mn} per Eq. 1.
        
        Returns:
            A^{mn}: (B, L_m, d_k)  –  cross-modal enriched representation
        """
        # Low-rank projections (Eq. 2)
        Q = self.W_Q(H_m)   # (B, L_m, d_k)
        K = self.W_K(H_n)   # (B, L_n, d_k)
        V = self.W_V(H_n)   # (B, L_n, d_k)

        # Scaled dot-product attention (Eq. 1)
        # scores ∈ (B, L_m, L_n); softmax applied row-wise over L_n
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)       # softmax_{L_n}
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        A_mn = torch.bmm(attn_weights, V)  # (B, L_m, d_k)
        return A_mn


class HCMA(nn.Module):
    """
    Hierarchical Cross-Modal Attention module.
    
    Stage 1 (Token-level): Pairwise cross-modal attention for all M(M-1)
        ordered pairs, with learnable combination weights α_{mn}.
    Stage 2 (Utterance-level): Mean-pool to utterance vectors, then
        gated fusion (Eq. 3-4).
    
    Complexity (Proposition 1):
        Attention interaction: O(M(M-1) L^2 d_k)
        Projection:           O(M(M-1) L d_in d_k)
        Gating:               O(M d^2)
    """

    def __init__(
        self,
        d_text: int = 768,
        d_audio: int = 512,
        d_visual: int = 256,
        d_k: int = 64,
        d_model: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.modality_names = ["text", "audio", "visual"]
        self.d_ins = {"text": d_text, "audio": d_audio, "visual": d_visual}

        # ── Projection layers to align all modalities to d_k ────────
        # Each modality may have different input dims, so we need
        # separate projectors for Q side and K/V side
        self.cross_attns = nn.ModuleDict()
        self.alpha_logits = nn.ParameterDict()  # learnable α_{mn}

        for m in self.modality_names:
            others = [n for n in self.modality_names if n != m]
            # Cross-attention modules for each (m, n) pair
            # Input is already projected to d_k by input_projs
            for n in others:
                key = f"{m}_to_{n}"
                self.cross_attns[key] = TokenLevelCrossModalAttention(
                    d_in=d_k,
                    d_k=d_k,
                    dropout=dropout,
                )
            # Learnable combination weights α_{mn}, softmaxed over M-1 (Eq. after Eq. 1)
            self.alpha_logits[m] = nn.Parameter(torch.zeros(len(others)))

        # ── Input projection: align each modality to common dim ─────
        self.input_projs = nn.ModuleDict({
            name: nn.Linear(dim, d_k) for name, dim in self.d_ins.items()
        })

        # ── Utterance-level projection: d_k -> d_model ──────────────
        self.utt_projs = nn.ModuleDict({
            name: nn.Linear(d_k, d_model) for name in self.modality_names
        })

        # ── Stage 2: Gated fusion (Eq. 3-4) ────────────────────────
        # W_g ∈ R^{d × 3d}, b_g ∈ R^d     (Eq. 3)
        self.W_g = nn.Linear(3 * d_model, d_model)
        # W_f ∈ R^{d × 3d}                  (Eq. 4)
        self.W_f = nn.Linear(3 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_text: torch.Tensor,    # (B, L_t, d_text)
        H_audio: torch.Tensor,   # (B, T_a, d_audio)
        H_visual: torch.Tensor,  # (B, T_v, d_visual)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through HCMA.
        
        Returns:
            z: (B, d_model) – fused utterance representation
            modality_utts: dict mapping modality name -> (B, d_model)
                           for MCS computation
        """
        inputs = {"text": H_text, "audio": H_audio, "visual": H_visual}

        # ── Project all modalities to common dim d_k ────────────────
        projected = {
            name: self.input_projs[name](inputs[name])
            for name in self.modality_names
        }

        # ── Stage 1: Token-level cross-modal attention ──────────────
        enriched = {}
        for m in self.modality_names:
            others = [n for n in self.modality_names if n != m]
            # Compute α weights via softmax over logits
            alpha = F.softmax(self.alpha_logits[m], dim=0)  # (M-1,)

            # Weighted sum of cross-attention outputs (Eq. after Eq. 1)
            H_tilde_m = torch.zeros_like(projected[m])
            for idx, n in enumerate(others):
                key = f"{m}_to_{n}"
                A_mn = self.cross_attns[key](projected[m], projected[n])
                H_tilde_m = H_tilde_m + alpha[idx] * A_mn

            enriched[m] = H_tilde_m  # (B, L_m, d_k)

        # ── Stage 2: Mean-pool to utterance vectors ─────────────────
        modality_utts = {}
        for m in self.modality_names:
            # Mean-pool over sequence length
            z_tilde = enriched[m].mean(dim=1)  # (B, d_k)
            # Project to d_model
            z_tilde = self.utt_projs[m](z_tilde)  # (B, d_model)
            modality_utts[m] = z_tilde

        # ── Gated fusion (Eq. 3-4) ─────────────────────────────────
        # Concatenate: [z_t; z_a; z_v] ∈ R^{3d}
        concat = torch.cat(
            [modality_utts["text"], modality_utts["audio"],
             modality_utts["visual"]],
            dim=-1,
        )  # (B, 3*d_model)

        # Gate vector (Eq. 3): g_i = σ(W_g [z_t; z_a; z_v] + b_g)
        g = torch.sigmoid(self.W_g(concat))  # (B, d_model)

        # Fused representation (Eq. 4): z_i = g_i ⊙ W_f [z_t; z_a; z_v]
        z = g * self.W_f(concat)  # (B, d_model)
        z = self.dropout(z)

        return z, modality_utts


# ─────────────────────────────────────────────────────────────────────
# 2. CASGT: Context-Aware Speaker Graph Transformer
#    Paper Section III-C
# ─────────────────────────────────────────────────────────────────────

class SpeakerGraph(nn.Module):
    """
    Constructs the speaker-relationship graph G = (V, E).
    
    Edge definition (Section III-C):
        e_{ij} = 1  if s_i == s_j               (intra-speaker)
        e_{ij} = 1  if s_i != s_j and |i-j| <= W (inter-speaker)
        e_{ij} = 0  otherwise
    
    Complexity: O(NW) edges vs O(N^2) for DialogueGCN (Table III).
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: context window W (default 10, Section IV-C)
        """
        super().__init__()
        self.W = window_size

    def build_adjacency(
        self,
        speaker_ids: torch.Tensor,  # (N,) speaker ID per utterance
        num_utterances: int,
    ) -> torch.Tensor:
        """
        Build sparse adjacency matrix.
        
        Returns:
            adj: (N, N) binary adjacency matrix
        """
        N = num_utterances
        adj = torch.zeros(N, N, device=speaker_ids.device)

        for i in range(N):
            for j in range(max(0, i - self.W), min(N, i + self.W + 1)):
                if i == j:
                    continue
                # Intra-speaker: always connect same speaker
                if speaker_ids[i] == speaker_ids[j]:
                    adj[i, j] = 1.0
                # Inter-speaker: connect if within window
                elif abs(i - j) <= self.W:
                    adj[i, j] = 1.0

        return adj


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer following Veličković et al. (2018).
    
    Computes: z_hat_i = Σ_{j ∈ N(i)} β_{ij} W_att z_j
    where β_{ij} are normalized attention coefficients (Section III-C).
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.3):
        super().__init__()
        self.W_att = nn.Linear(d_in, d_out, bias=False)
        self.a = nn.Parameter(torch.empty(2 * d_out))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,     # (N, d_in) node features
        adj: torch.Tensor,   # (N, N) adjacency
    ) -> torch.Tensor:
        """Compute graph-attention-weighted node features."""
        N = z.size(0)
        h = self.W_att(z)  # (N, d_out)

        # Compute attention coefficients
        # a^T [h_i || h_j] for all pairs
        h_i = h.unsqueeze(1).expand(N, N, -1)   # (N, N, d_out)
        h_j = h.unsqueeze(0).expand(N, N, -1)   # (N, N, d_out)
        e = self.leaky_relu(
            torch.sum(self.a * torch.cat([h_i, h_j], dim=-1), dim=-1)
        )  # (N, N)

        # Mask non-adjacent pairs
        e = e.masked_fill(adj == 0, float("-inf"))
        beta = F.softmax(e, dim=-1)       # β_{ij}
        beta = self.dropout(beta)
        beta = beta.masked_fill(adj == 0, 0.0)

        # Aggregate
        z_hat = torch.mm(beta, h)  # (N, d_out)
        return z_hat


class CASGT(nn.Module):
    """
    Context-Aware Speaker Graph Transformer.
    
    Architecture (Section III-C):
        1. Build speaker graph with window W
        2. Apply graph attention to get β-weighted representations
        3. Feed through 4-layer, 8-head transformer encoder
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        window_size: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.graph = SpeakerGraph(window_size)
        self.gat = GraphAttentionLayer(d_model, d_model, dropout)

        # 4-layer transformer encoder with 8 heads (Section IV-C)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,           # (B, N, d_model) fused utterance features
        speaker_ids: torch.Tensor,  # (B, N) speaker IDs
    ) -> torch.Tensor:
        """
        Process conversation through graph attention + transformer.
        
        Returns:
            z_hat: (B, N, d_model) context-enriched representations
        """
        B, N, D = z.shape
        outputs = []

        for b in range(B):
            # Build adjacency for this conversation
            adj = self.graph.build_adjacency(speaker_ids[b], N)
            # Graph attention
            z_gat = self.gat(z[b], adj)  # (N, d_model)
            outputs.append(z_gat)

        z_gat = torch.stack(outputs, dim=0)  # (B, N, d_model)

        # Residual connection + transformer
        z_gat = self.norm(z_gat + z)
        z_hat = self.transformer(z_gat)  # (B, N, d_model)

        return z_hat


# ─────────────────────────────────────────────────────────────────────
# 3. MCS: Modality Contribution Score
#    Paper Section III-E, Equations 5-6
# ─────────────────────────────────────────────────────────────────────

class MCSLayer(nn.Module):
    """
    Modality Contribution Score layer.
    
    Prediction (Eq. 5):
        y_hat = softmax(W_c @ z_hat + b_c)
    
    Attribution (Eq. 6):
        m_i^k = ||W_c^{(k)} @ z_tilde_i^k||_2^2 / 
                Σ_{k'} ||W_c^{(k')} @ z_tilde_i^{k'}||_2^2
    
    where W_c^{(k)} is the classifier weight block for modality k's
    contribution dimensions. This decomposes prediction energy across
    modalities (Eq. 7 justification).
    
    Note: MCS is an APPROXIMATE attribution (Section III-E, "Interaction
    Effects and Approximation Scope"). Interaction terms from gating/CASGT
    are not captured.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_classes: int = 6,
        num_modalities: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_modalities = num_modalities

        # Classifier: W_c ∈ R^{|Y| × d}, b_c ∈ R^{|Y|}
        self.classifier = nn.Linear(d_model, num_classes)

        # Per-modality projection heads for MCS energy computation
        # Each maps modality utterance vector to logit space
        self.mod_projectors = nn.ModuleDict({
            "text": nn.Linear(d_model, num_classes, bias=False),
            "audio": nn.Linear(d_model, num_classes, bias=False),
            "visual": nn.Linear(d_model, num_classes, bias=False),
        })

    def forward(
        self,
        z_hat: torch.Tensor,                    # (B, d_model) CASGT output
        modality_utts: Dict[str, torch.Tensor],  # {name: (B, d_model)}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prediction and modality contribution scores.
        
        Returns:
            logits:    (B, num_classes)
            mcs_scores: (B, 3) – per-modality contribution scores summing to 1
        """
        # ── Prediction (Eq. 5) ──────────────────────────────────────
        logits = self.classifier(z_hat)  # (B, num_classes)

        # ── Attribution (Eq. 6) ─────────────────────────────────────
        # For each modality k, compute energy:
        #   ||W_c^{(k)} @ z_tilde^k||_2^2
        # using learned per-modality projectors
        energies = []
        for name in ["text", "audio", "visual"]:
            # z_tilde^k = modality k's utterance representation
            z_k = modality_utts[name]  # (B, d_model)

            # Project to logit space: W_c^{(k)} @ z_tilde^k
            projection = self.mod_projectors[name](z_k)  # (B, num_classes)

            # Energy = ||projection||_2^2
            energy = (projection ** 2).sum(dim=-1)  # (B,)
            energies.append(energy)

        # Stack energies: (B, 3)
        energies = torch.stack(energies, dim=-1)

        # Normalize to get MCS scores (Eq. 6): sum_k m_i^k = 1
        mcs_scores = energies / (energies.sum(dim=-1, keepdim=True) + 1e-8)

        return logits, mcs_scores


# ─────────────────────────────────────────────────────────────────────
# 4. Full IMFER Framework
#    Paper Section III-A
# ─────────────────────────────────────────────────────────────────────

class IMFER(nn.Module):
    """
    IMFER: Interpretable Multimodal Fusion for Emotion Recognition.
    
    Architecture (Fig. 1):
        1. Modality-specific encoders (RoBERTa, wav2vec2, 3D-ResNet)
        2. HCMA: token-level cross-modal attention + gated fusion
        3. CASGT: speaker graph transformer
        4. MCS: prediction + modality attribution
    
    Note: In this implementation, encoders are assumed to be run
    externally (features pre-extracted). This module takes encoder
    outputs as input.
    """

    def __init__(
        self,
        d_text: int = 768,
        d_audio: int = 512,
        d_visual: int = 256,
        d_k: int = 64,
        d_model: int = 512,
        num_classes: int = 6,
        casgt_heads: int = 8,
        casgt_layers: int = 4,
        context_window: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model

        # ── HCMA module (Section III-D) ─────────────────────────────
        self.hcma = HCMA(
            d_text=d_text,
            d_audio=d_audio,
            d_visual=d_visual,
            d_k=d_k,
            d_model=d_model,
            dropout=dropout,
        )

        # ── CASGT module (Section III-C) ────────────────────────────
        self.casgt = CASGT(
            d_model=d_model,
            num_heads=casgt_heads,
            num_layers=casgt_layers,
            window_size=context_window,
            dropout=dropout,
        )

        # ── MCS + Classifier (Section III-E) ────────────────────────
        self.mcs = MCSLayer(
            d_model=d_model,
            num_classes=num_classes,
            num_modalities=3,
        )

    def forward(
        self,
        H_text: torch.Tensor,       # (B, L_t, d_text) per-utterance
        H_audio: torch.Tensor,      # (B, T_a, d_audio)
        H_visual: torch.Tensor,     # (B, T_v, d_visual)
        speaker_ids: torch.Tensor,  # (B_conv, N) for CASGT
        utterance_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        For simplicity, this processes utterances independently through
        HCMA, then batches them into conversations for CASGT.
        
        Returns dict with:
            - logits:     (B, num_classes)
            - mcs_scores: (B, 3)
            - modality_utts: dict of per-modality utterance vectors
        """
        # ── Step 1: HCMA fusion ─────────────────────────────────────
        z_fused, modality_utts = self.hcma(H_text, H_audio, H_visual)
        # z_fused: (B, d_model)

        # ── Step 2: CASGT (simplified: treat batch as one conversation)
        # In practice, you'd reshape per-conversation
        z_conv = z_fused.unsqueeze(0)        # (1, B, d_model)
        spk = speaker_ids.unsqueeze(0) if speaker_ids.dim() == 1 else speaker_ids
        z_hat = self.casgt(z_conv, spk)      # (1, B, d_model)
        z_hat = z_hat.squeeze(0)             # (B, d_model)

        # ── Step 3: MCS prediction + attribution ────────────────────
        logits, mcs_scores = self.mcs(z_hat, modality_utts)

        return {
            "logits": logits,
            "mcs_scores": mcs_scores,
            "modality_utts": modality_utts,
            "z_fused": z_fused,
            "z_hat": z_hat,
        }


# ─────────────────────────────────────────────────────────────────────
# 5. Utility: Parameter counting
# ─────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = IMFER(num_classes=6)
    B, L_t, T_a, T_v = 4, 50, 30, 16
    H_text = torch.randn(B, L_t, 768)
    H_audio = torch.randn(B, T_a, 512)
    H_visual = torch.randn(B, T_v, 256)
    speaker_ids = torch.randint(0, 2, (B,))

    out = model(H_text, H_audio, H_visual, speaker_ids)
    print(f"Logits shape:     {out['logits'].shape}")
    print(f"MCS scores shape: {out['mcs_scores'].shape}")
    print(f"MCS sum (≈1):     {out['mcs_scores'].sum(dim=-1)}")
    print(f"Total parameters: {count_parameters(model):,}")
