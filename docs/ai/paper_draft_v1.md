# Title: Causal Graph Neural Networks for Robust Multimodal Disaster Classification: A Three-Level Disentanglement Framework

## Abstract
Social media has become incredibly important for disaster response, but multimodal classification models often suffer from "spurious correlations," relying on domain-specific biases rather than genuine disaster indicators. We propose **CausalCrisis v2**, a novel framework for multimodal disaster classification that achieves state-of-the-art out-of-distribution (OOD) generalization. Our framework introduces a **Three-Level Disentanglement** architecture: (1) extracting modal-general semantics from visual and textual modalities, (2) separating causal disaster features from spurious domain noise using Gradient Reversal Layers (GRL), and (3) unifying causal features within a Soft-Attention K-Nearest Neighbor (k-NN) Graph. To further eliminate bias, we augment our graph layer with **Vector-Space Backdoor Adjustment** and a novel **Edge Heterophily Scorer** that dynamically penalizes out-of-distribution spurious edges based on feature and pseudo-label dissimilarities. Extensive experiments on the CrisisMMD v2.0 dataset reveal that CausalCrisis v2 significantly outperforms existing methodologies, particularly in challenging, heavily imbalanced tasks like Humanitarian Categories.

## 1. Introduction
- Importance of computational models for social media disaster management.
- The failure of current Multimodal models (CLIP, MLP-based approaches) due to spurious correlations (e.g., associating a dark sky inherently with a hurricane instead of genuine damage).
- Current DG (Domain Generalization) models often lack structural reasoning and are confined to flat MLP architectures.
- **Contributions**:
  1. A graph-enhanced causal reasoning pipeline for multimodal Domain Generalization.
  2. Integration of Vector-Space Backdoor Adjustment combined with Soft-Attention GAT (Graph Attention Network) applied specifically on Causal Features.
  3. A dynamic **Edge Heterophily Scorer** that penalizes spurious connections (heterophilic edges) between nodes using ground-truth and pseudo-labels, significantly improving OOD robustness.
  4. A robust Three-Level Disentanglement framework showing state-of-the-art Balanced Accuracy in severe class-imbalance tasks.

## 2. Related Work
- **Multimodal Disaster Classification**: Early works relying on traditional CNNs/RNNs, shift towards Transformer-based (CLIP, BERT) backbones. Limitations on In-Distribution overfitting.
- **Causal Inference in Deep Learning**: Introduction of Gradient Reversal Layers (GRL) and Front-door/Back-door adjustments. Discussing CIRL, CAL, CAMO implementations.
- **Graph Neural Networks for Causal Reasoning**: How graph structures capture relational and contextual cues that flat MLPs fail to encode.

## 3. Proposed Methodology: CausalCrisis v2
### 3.1. Stage 1: Multimodal Feature Extraction
- Employing Frozen CLIP (ViT-L/14) for both textual and visual modalities to leverage strong pre-aligned semantic foundations.

### 3.2. Stage 2: Modality Disentanglement
- Dual-projection architecture splitting raw features into *Modal-General* ($z_m$) and *Modal-Specific* ($\tilde{z}_m$) spaces. 
- Regularization via Supervised Contrastive Loss (SupCon) and Orthogonal Cosine Penalty.

### 3.3. Stage 3: Causal Disentanglement via GRL
- Fusing general features into a unified embedding.
- MLP-based bifurcations into Causal Features ($X_c$) and Spurious Features ($X_s$).
- Using a Gradient Reversal Layer over $X_c$ attempting to fool the Domain Classifier, forcing $X_c$ to discard domain-specific footprints.

### 3.4. Stage 4: Graph-Enhanced Causal Reasoning & Backdoor Adjustment
- **Dynamic Soft-Attention K-NN Graph**: Instead of hard-edges, we use a Softmax-weighted adjacency matrix over top-k causal neighbors ($T=0.1, K=5$).
- **Vector-Level Backdoor Adjustment**: Instead of an independent linear layer, we merge graph-enhanced features with historical spurious features ($X_{gnn} + X_s$) to enforce prediction stability, mathematically approximating $P(Y | do(X_c))$.

### 3.5. Stage 5 (Phase 3c): Heterophily-Aware Graph Construction
- **Spurious Edge Penalty**: Standard k-NN graphs suffer from connecting nodes solely based on spatial proximity, which sometimes links samples with similar spurious embeddings but fundamentally opposing semantics (Heterophily). 
- **Dynamic Edge Scorer**: We deploy an `EdgeHeterophilyScorer` that evaluates the causal feature disparity ($|x_i - x_j|$) and label similarity ($p_i \cdot p_j$).
- **OOD Graph Pruning**: During IID training, ground-truth distributions train the scorer. During OOD inference, Phase 1 logits act as pseudo-labels. Edges identified as spurious receive massive logarithmic penalties before Softmax attention, practically cutting off destructive message-passing lanes.

## 4. Experiments and Results
### 4.1. Dataset & Setup
- CrisisMMD v2.0, formatted for 3 Tasks (Informative, Humanitarian, Severity).
- Leave-One-Disaster-Out (LODO) and global Multi-seed evaluation configurations (seeds 42, 100, 2026).

### 4.2. Results: Task 1 (Informative - 2 Classes)
- **Results:** F1 ~ 0.7947, bAcc ~ 0.7951.
- *Analysis:* The model proves that the complex causal pipeline does not degrade binary task performance, maintaining highly robust accuracy.

### 4.3. Results: Task 2 (Humanitarian - 8 Classes)
- **Results:** F1 ~ 0.6106, bAcc ~ 0.6229.
- *Analysis:* The crown jewel of our framework. Overcoming extreme class imbalances, the model vastly outperforms the ~42% baseline, validating the efficacy of Graph + Backdoor Adjustment against catastrophic forgetting of minority classes.

### 4.4. Results: Task 3 (Severity - 3 Classes)
- **Results:** F1 ~ 0.6960, bAcc ~ 0.7108.
- *Analysis:* High stability in severity reasoning.

### 4.5. Ablation Study
- Comparison between Phase 1 (Pure MLP) vs. Phase 2 (Soft-Attention GNN + BA).
- Analyzing the devastating effects of *Hard-Edge Over-smoothing* vs. the success of *Soft-Attention*.
- The necessity of Vector Addition over external Linear layers in Backdoor Adjustment.

### 4.6. Analyzing the Heterophily Scorer
- Evaluating the impact of dynamic edge penalties on the Soft-Attention graphs.
- Visualizing graph connections with and without the Heterophily Scorer during pure OOD evaluations to witness how pseudo-labels successfully prune detrimental semantic bridges.

## 5. Conclusion
- Summary of the Three-Level architecture.
- Reaffirming the SOTA results on CrisisMMD.
- Future works: Scaling graph sizes and scaling the causal concept up to video or temporal data.
