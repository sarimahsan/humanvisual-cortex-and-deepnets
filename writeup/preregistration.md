# Pre-Registered Analysis Plan and Hypotheses

**Project Title**: Brain encoding project: which deep-net layers explain which human visual regions?  
**Date of Pre-Registration**: October 2026 (Committed prior to model fitting and inspection of experimental results)  
**Target Dataset**: Algonauts 2023 Challenge release of the Natural Scenes Dataset (NSD)  
**Primary Subjects**: Subjects 1–4 (evaluated independently for multi-subject robustness)  

---

## 1. Study Overview & Scientific Scope

This study investigates functional alignment between the hierarchical representations of deep neural networks (DNNs) and the human visual cortex using 7T fMRI responses to natural scenes. 

### Non-Goals
To maintain strict methodological honesty:
- **No causal or mechanistic claims**: High encoding accuracy indicates statistical alignment and representational similarity under linear readout, not that the biological brain implements backpropagation, convolutions, or transformer self-attention.
- **No architectural novelty**: We use standard, publicly available, pre-trained weights to isolate representational characteristics.
- **No decoding or image reconstruction**: The analysis is strictly forward encoding ($f(\text{image}) \to \text{fMRI responses}$).
- **No leaderboard tuning**: No post-hoc layer selection or hyperparameter tuning on the evaluation test split.

---

## 2. Pre-Registered Hypotheses ($H_1$ – $H_5$)

### $H_1$: Hierarchical Correspondence Across Cortical Streams
* **Prediction**: As we move anatomically from early retinotopic visual cortex (V1, V2, V3) to intermediate areas (hV4) and high-level ventral temporal cortex (category-selective ROIs: FFA, PPA, EBA), the layer index providing maximal linear encoding accuracy will monotonically increase.
* **Metric**: Rank correlation (Spearman's $\rho$) between anatomical ROI order and peak layer normalized depth $[0, 1]$.
* **Expected Result**: $\rho > 0$ with $p < 0.01$ across all trained feedforward and vision transformer models.

### $H_2$: Learned Representations vs. Random Inductive Bias
* **Prediction**: A randomly initialized (untrained) ResNet-50 will exhibit non-zero encoding performance in early visual areas (due to random convolutional filter spatial biases acting as edge detectors), but will severely underperform trained ResNet-50 in intermediate and high-level ROIs.
* **Expected Result**: The performance gap $\Delta r = r_{\text{trained}} - r_{\text{untrained}}$ will be smallest in V1 and largest in category-selective areas (FFA, PPA, EBA).

### $H_3$: Impact of Training Objective (Controlled Architecture)
* **Prediction**: Controlling for architecture using Vision Transformers of identical capacity (ViT-S/16), self-supervised pre-training (DINO) and language-image pre-training (CLIP) will capture higher-level category-selective representations better than or distinctively from standard supervised ImageNet classification (DeiT-S).
* **Direct Comparison**: Paired bootstrap difference on held-out test images between DeiT-S and DINO ViT-S/16.

### $H_4$: Pixel and Low-Level Spatial Frequency Controls
* **Prediction**: A classical multiscale Gabor wavelet pyramid with color and spatial-frequency statistics will explain substantial variance in V1 and V2, comparable to early DNN layers, but will approach near-zero prediction accuracy in high-level ventral regions.

### $H_5$: Cross-Subject Robustness
* **Prediction**: While raw correlation values will vary across individual human subjects due to differences in fMRI head motion, wakefulness, and signal-to-noise ratio (SNR), the **relative hierarchy rank ($\rho$)** and **model performance rankings** will be conserved across Subjects 1 through 4.

---

## 3. Strict Methodological Controls & Leakage Prevention

1. **Held-Out Test Set**: 
   - A deterministic held-out test split (15% of images, ~1,300 images per subject) is partitioned *before* any feature transformation.
   - Test images are never used for PCA fitting, feature standardization, or Ridge alpha parameter selection.
2. **Standardization & PCA**:
   - For all layers and models, PCA is fit strictly on the training partition.
   - All layers are reduced to a matched dimensionality of $K = 1024$ principal components to prevent higher-dimensional layers from having an unfair regression advantage.
3. **Hyperparameter Selection (Inner Cross-Validation)**:
   - Ridge regularization parameter $\alpha$ is chosen via inner 5-fold cross-validation over a log grid: $\alpha \in [10^{-1}, 10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6]$.
   - Hyperparameters are selected per ROI group using closed-form GPU SVD / kernel formulations.
4. **Permutation / Shuffled Null Distribution**:
   - For each subject and ROI, a null baseline is computed by randomly permuting image identities relative to fMRI responses prior to model fitting.
   - The null distribution must center at Pearson $r \approx 0 \pm 0.02$.
5. **Statistical Significance & Confidence Intervals**:
   - All correlations are evaluated as Pearson $r$ between measured and predicted responses across held-out images.
   - Summary statistics per ROI are computed as the median across vertices within the ROI, accompanied by 95% bootstrap confidence intervals resampled over test images (1,000 bootstrap iterations).
