# IMFER: Interpretable Multimodal Fusion for Emotion Recognition

Official implementation of **"IMFER: An Interpretable Multimodal Fusion Framework for Emotion Recognition in Conversational AI via Cross-Modal Attention and Explainability Mechanisms"**

## Architecture

IMFER comprises four components:
1. **Modality-Specific Encoders**: RoBERTa-base (text), wav2vec 2.0 (audio), 3D-ResNet (visual)
2. **HCMA**: Hierarchical Cross-Modal Attention with low-rank projection (92% attention FLOPs reduction)
3. **CASGT**: Context-Aware Speaker Graph Transformer with windowed graph (O(NW) vs O(N²))
4. **MCS**: Modality Contribution Score layer for ante-hoc interpretability

## Results

| Dataset  | WF1 (%)         | Acc (%) | #Params |
|----------|-----------------|---------|---------|
| IEMOCAP  | 69.87 ± 0.21    | 68.93   | 59.4M   |
| MELD     | 62.34 ± 0.18    | 63.17   | 59.4M   |
| EmoryNLP | 40.21 ± 0.29    | —       | 59.4M   |

## Project Structure

```
src/
├── config.py                 # All hyperparameters (Section IV-C)
├── models.py                 # HCMA, CASGT, MCS, full IMFER (Section III)
├── losses.py                 # L_CE + λ₁·L_MCS + λ₂·L_align (Eq. 7-9)
├── train.py                  # Training loop with 5-seed evaluation
├── evaluate.py               # WF1, MF1, AOPC, statistical tests (Section V)
├── complexity_analysis.py    # Proposition 1 FLOPs derivation (Section III-D)
├── robustness.py             # Noise injection + missing modality (Section V-G)
└── visualize_results.py      # All paper figures (Figs. 3-12)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (demo with synthetic data)
cd src
python train.py --dataset iemocap --device cpu

# Generate all figures
python visualize_results.py

# Verify FLOPs analysis (Proposition 1)
python complexity_analysis.py

# Run robustness analysis
python robustness.py

# Run AOPC evaluation
python evaluate.py
```

## Key Equations

| Equation | Description | File |
|----------|-------------|------|
| Eq. 1-2  | Token-level cross-modal attention | `models.py::TokenLevelCrossModalAttention` |
| Eq. 3-4  | Gated utterance fusion | `models.py::HCMA` |
| Eq. 5    | Classifier prediction | `models.py::MCSLayer` |
| Eq. 6    | MCS attribution scores | `models.py::MCSLayer` |
| Eq. 7    | Combined loss | `losses.py::IMFERLoss` |
| Eq. 8    | MCS entropy regularizer | `losses.py::MCSEntropyLoss` |
| Eq. 9    | Contrastive alignment | `losses.py::ContrastiveAlignmentLoss` |
| Eq. 10   | Attention cost ratio | `complexity_analysis.py` |
| Prop. 1  | HCMA complexity | `complexity_analysis.py::compute_hcma_flops` |

## Hardware

All experiments conducted on Apple M4 Pro (12-core CPU, 20-core GPU, 48 GB unified memory) with PyTorch 2.2 MPS backend.

## Citation

```bibtex
@article{sathalla2025imfer,
  title={IMFER: An Interpretable Multimodal Fusion Framework for Emotion 
         Recognition in Conversational AI via Cross-Modal Attention and 
         Explainability Mechanisms},
  author={Sathalla, Suresh and Babu, Pandava Sudharshan and 
          Ramegowda, Mahesh and Gottam, Omprakash},
  year={2025}
}
```

## License

This project is for academic research purposes.
