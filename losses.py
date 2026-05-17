"""
losses.py – Training objectives for IMFER.

Implements:
  - L_CE:    Cross-entropy classification loss
  - L_MCS:   Entropy-maximization MCS regularizer (Eq. 8)
  - L_align: Contrastive text-audio alignment loss (Eq. 9)
  - L_total: Combined objective (Eq. 7)

Equation numbers reference the paper's Section III-F.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCSEntropyLoss(nn.Module):
    """
    MCS entropy-maximization regularizer (Eq. 8).
    
    L_MCS = -(1/N) Σ_i Σ_k  m_i^k  log(m_i^k + ε)
    
    This MAXIMIZES the entropy of MCS distributions, pushing the model
    toward balanced modality usage and preventing modality collapse.
    The negative sign means minimizing this loss = maximizing entropy.
    
    Hyperparameter: λ_1 = 0.1 (Section IV-C, Fig. 9 sensitivity)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mcs_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mcs_scores: (B, M) modality contribution scores, Σ_k m^k = 1
        
        Returns:
            Scalar entropy loss (to be MINIMIZED; minimizing this = maximizing entropy)
        """
        # Entropy: H(m) = -Σ_k m^k log(m^k)
        # Loss = -H(m) = Σ_k m^k log(m^k)  (negative entropy)
        # When we minimize this, we maximize entropy
        log_mcs = torch.log(mcs_scores + self.eps)
        entropy = -(mcs_scores * log_mcs).sum(dim=-1)  # (B,)
        # We want to MAXIMIZE entropy, so loss = -mean(entropy)
        return -entropy.mean()


class ContrastiveAlignmentLoss(nn.Module):
    """
    Contrastive text-audio alignment loss (Eq. 9).
    
    L_align = -(1/N) Σ_i log[ exp(sim(z_t^i, z_a^i)/τ) / 
                                Σ_j exp(sim(z_t^i, z_a^j)/τ) ]
    
    This aligns text and audio representations in the shared space.
    Only text-audio alignment is used (not text-visual or audio-visual)
    because:
      1. Text and audio are the two dominant modalities (Section III-F)
      2. Adding more pairs didn't improve WF1 (+0.08%, not significant)
    
    Hyperparameters: τ = 0.07, λ_2 = 0.05 (Section IV-C)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        z_text: torch.Tensor,   # (B, d_model) text utterance vectors
        z_audio: torch.Tensor,  # (B, d_model) audio utterance vectors
    ) -> torch.Tensor:
        """
        Compute InfoNCE-style contrastive loss between text and audio.
        
        Returns:
            Scalar alignment loss
        """
        # Normalize embeddings
        z_t = F.normalize(z_text, dim=-1)    # (B, d)
        z_a = F.normalize(z_audio, dim=-1)   # (B, d)

        # Similarity matrix: sim(z_t^i, z_a^j) for all (i, j)
        sim_matrix = torch.mm(z_t, z_a.T) / self.tau  # (B, B)

        # Positive pairs are on the diagonal (same utterance)
        labels = torch.arange(z_t.size(0), device=z_t.device)

        # Cross-entropy over similarity scores (Eq. 9)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class IMFERLoss(nn.Module):
    """
    Combined IMFER training objective (Eq. 7).
    
    L = L_CE + λ_1 * L_MCS + λ_2 * L_align
    
    where:
      - L_CE:    standard cross-entropy for emotion classification
      - L_MCS:   entropy regularizer to prevent modality collapse
      - L_align: contrastive text-audio alignment
      - λ_1 = 0.1, λ_2 = 0.05  (Section IV-C)
    
    The tension between L_CE (pushes toward most discriminative modality
    combination) and L_MCS (pushes toward diversity) is a key design
    feature (Section III-F, paragraph on tension).
    """

    def __init__(
        self,
        num_classes: int = 6,
        lambda_1: float = 0.1,
        lambda_2: float = 0.05,
        tau: float = 0.07,
        class_weights: torch.Tensor = None,
    ):
        """
        Args:
            num_classes:   number of emotion classes
            lambda_1:      MCS entropy weight (Eq. 7)
            lambda_2:      alignment weight (Eq. 7)
            tau:           temperature for alignment (Eq. 9)
            class_weights: optional per-class weights for imbalanced datasets
                           (MELD uses weighted CE, Section IV-A)
        """
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # L_CE: Cross-entropy (optionally weighted for MELD)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        # L_MCS: Entropy regularizer (Eq. 8)
        self.mcs_loss = MCSEntropyLoss()
        # L_align: Contrastive alignment (Eq. 9)
        self.align_loss = ContrastiveAlignmentLoss(temperature=tau)

    def forward(
        self,
        logits: torch.Tensor,      # (B, num_classes) model predictions
        labels: torch.Tensor,      # (B,) ground-truth emotion labels
        mcs_scores: torch.Tensor,  # (B, M) modality contribution scores
        z_text: torch.Tensor,      # (B, d) text utterance vectors
        z_audio: torch.Tensor,     # (B, d) audio utterance vectors
    ) -> dict:
        """
        Compute combined loss (Eq. 7).
        
        Returns:
            dict with 'total', 'ce', 'mcs', 'align' loss values
        """
        # L_CE: classification loss
        l_ce = self.ce_loss(logits, labels)

        # L_MCS: entropy regularizer (Eq. 8)
        l_mcs = self.mcs_loss(mcs_scores)

        # L_align: contrastive alignment (Eq. 9)
        l_align = self.align_loss(z_text, z_audio)

        # Combined (Eq. 7)
        l_total = l_ce + self.lambda_1 * l_mcs + self.lambda_2 * l_align

        return {
            "total": l_total,
            "ce": l_ce,
            "mcs": l_mcs,
            "align": l_align,
        }


if __name__ == "__main__":
    # Sanity check
    B, C = 8, 6
    logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))
    mcs = F.softmax(torch.randn(B, 3), dim=-1)
    z_t = torch.randn(B, 512)
    z_a = torch.randn(B, 512)

    criterion = IMFERLoss(num_classes=C)
    losses = criterion(logits, labels, mcs, z_t, z_a)

    for k, v in losses.items():
        print(f"L_{k:6s} = {v.item():.4f}")
