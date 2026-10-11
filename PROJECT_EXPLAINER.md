# Project Explainer: Human Visual Cortex & Deep Neural Network Encoding

A comprehensive conceptual guide to the computational methodology, neuroscience principles, and experimental results of this brain encoding project.

---

## 1. Executive Summary & Research Motivation

### What is Brain Encoding?
In computational cognitive neuroscience, a **brain encoding model** is a predictive computational framework that models how sensory stimuli (such as natural photographs) are transformed into biological neural activity in the brain.

```
       Natural Visual Stimulus (Natural Scene Image)
                    /                     \
                   v                       v
    Biological Visual Cortex       Deep Neural Network
      (7T fMRI BOLD Signal)      (Layer Activations: CNN/ViT)
                   \                       /
                    v                     v
          Voxel-wise Linear Encoding Model: W
        Predicted fMRI Activity vs. Measured fMRI Activity
```

While traditional neuroscience often used simplified, artificial stimuli (e.g., drifting gratings or isolated geometric shapes), modern encoding models leverage massive datasets of natural photographs combined with high-resolution functional Magnetic Resonance Imaging (fMRI) to study the visual system in ecologically valid conditions.

### The Central Scientific Questions
1. **Hierarchical Alignment**: Does the representational hierarchy of artificial deep neural networks correspond to the anatomical hierarchy of the primate visual system? Specifically, do early network layers best predict early visual cortex ($V1$–$V3$), while late layers best predict category-selective visual areas ($FFA$, $PPA$, $RSC$)?
2. **Objective vs. Architecture**: How do supervised vision models, self-supervised networks, and multimodal vision-language models (e.g., CLIP) compare in predicting biological visual representations when architecture is controlled?
3. **Task-Driven Learning vs. Inductive Bias**: Does biological hierarchy alignment arise from visual task optimization (learning), or does an untrained convolutional architecture with random weights exhibit similar hierarchical alignment purely due to convolutional inductive biases?

---

## 2. Experimental Data & Neuroimaging Foundation

### The Natural Scenes Dataset (NSD)
This project uses data from the **Natural Scenes Dataset (NSD)** via the Algonauts 2023 Challenge release:
* **Imaging Modality**: Ultra-high-field **7-Tesla (7T) fMRI**, providing high signal-to-noise ratio and millimeter-scale spatial resolution.
* **Stimulus Set**: Thousands of high-resolution natural color photographs from the Microsoft COCO dataset depicting complex everyday scenes (people, animals, objects, indoor and outdoor environments).
* **Cortical Measurements**: High-resolution cortical surface reconstructions covering **39,548 vertices** across both the left and right hemispheres of the visual cortex.

### The Primate Visual Hierarchy
The human visual system processes visual inputs along a ventral ("what") and dorsal ("where/how") pathway characterized by increasing receptive field size and representational complexity:

| Cortical Area | Visual Processing Stage | Functional Role & Tuning |
|:---|:---:|:---|
| **V1** (Primary Visual Cortex) | Early Retinotopic | Local contrast, oriented edges, spatial frequencies. |
| **V2 & V3** | Early Visual | Angles, texture boundaries, intermediate contours, binocular disparity. |
| **hV4** | Intermediate Form | Color constancy, curvature, 2D shape combinations. |
| **OFA** (Occipital Face Area) | Mid-Level Category | Face components (eyes, nose, mouth) prior to holistic integration. |
| **OPA** (Occipital Place Area) | Mid-Level Scene | Navigational boundaries, local geometry of scene surfaces. |
| **EBA** (Extrastriate Body Area) | High-Level Category | Human bodies, body parts, and silhouettes (excluding faces). |
| **FFA** (Fusiform Face Area) | High-Level Ventral | Holistic face identity and face perception. |
| **PPA** (Parahippocampal Place Area) | High-Level Ventral | Spatial layout, indoor/outdoor scenes, buildings, landscapes. |
| **RSC** (Retrosplenial Complex) | High-Level Ventral/Medial | Scene memory, spatial orientation, large-scale environmental context. |

---

## 3. Computational Models & Architectural Paradigms

To systematically evaluate which computational representations best model the visual cortex, this benchmark compares five model configurations:

1. **AlexNet (Supervised CNN Baseline)**:
   - 8-layer convolutional neural network trained on ImageNet-1k.
   - Serves as the classic anchor benchmark in deep learning vision and computational neuroscience.
2. **ResNet-50 (Modern Deep CNN)**:
   - 50-layer residual network with skip connections, trained on ImageNet-1k.
   - Evaluated across 8 standardized layer stages spanning early stem convolutions to late bottleneck blocks.
3. **DeiT-S / ViT-S/16 (Vision Transformer)**:
   - Self-attention architecture based on non-overlapping $16 \times 16$ image patches without spatial convolutions.
   - Evaluated across transformer stages to test whether self-attention recapitulates ventral stream hierarchy.
4. **CLIP ViT-B/16 (Multimodal Vision-Language)**:
   - Transformer visual encoder trained via contrastive language-image pre-training on 400M image-text pairs.
   - Tests whether semantic language alignment enhances predictivity in high-level visual cortex.
5. **Untrained ResNet-50 (Architectural Control)**:
   - Identical architecture and depth to ResNet-50, initialized with random Gaussian weights without training.
   - Serves as an essential negative control to isolate learned features from architectural inductive bias.
6. **Gabor Wavelet Pyramid (Classical Baseline)**:
   - Multi-scale, multi-orientation Gabor filter bank simulating classical textbook receptive fields of simple and complex cells.

---

## 4. End-to-End Encoding Pipeline

The computational pipeline implements a linear encoding workflow:

```
  [Stimulus Images]
         |
         v
  [1. Feature Extraction]  --> Hook activations across all model stages
         |
         v
  [2. Spatial Pooling]     --> Standardized spatial aggregation (e.g. AdaptiveAvgPool2d)
         |
         v
  [3. PCA Reduction]       --> Fit PCA on training stimuli (1,024 dimensions)
         |
         v
  [4. Ridge Regression]    --> Fit L2-regularized linear model on training split
         |
         v
  [5. Out-of-Sample Eval]  --> Predict responses for held-out test images
         |
         v
  [6. Cross-Validation]    --> Vertex-wise Pearson correlation (r) & Hierarchy tests
```

### Mathematical Formulation

#### 1. Feature Preprocessing & PCA
For image stimulus $i$, model layer $l$ outputs activation tensor $\mathbf{A}_l(i)$. After spatial adaptive pooling, activations are flattened into feature vector $\mathbf{x}_l(i) \in \mathbb{R}^D$. To prevent overfitting and standardize dimensionality across architectures, Principal Component Analysis (PCA) is fit strictly on training set stimuli to retain the top $K = 1,024$ principal components:

$$\mathbf{Z}_{\text{train}} = \mathbf{X}_{\text{train}} \mathbf{V}_K, \quad \mathbf{Z}_{\text{test}} = \mathbf{X}_{\text{test}} \mathbf{V}_K$$

#### 2. Voxel-wise Ridge Regression
For feature matrix $\mathbf{Z} \in \mathbb{R}^{N \times K}$ and measured cortical fMRI responses $\mathbf{Y} \in \mathbb{R}^{N \times V}$ across $V$ vertices:

$$\hat{\mathbf{W}} = \arg\min_{\mathbf{W}} \|\mathbf{Y} - \mathbf{Z}\mathbf{W}\|_2^2 + \alpha \|\mathbf{W}\|_2^2 = (\mathbf{Z}^T \mathbf{Z} + \alpha \mathbf{I})^{-1} \mathbf{Z}^T \mathbf{Y}$$

The regularization hyperparameter $\alpha$ is selected via cross-validation over a logarithmic grid $\alpha \in \{10^1, 10^2, \dots, 10^7\}$.

#### 3. Prediction Accuracy ($r$)
Model performance on the held-out test split is evaluated using the Pearson correlation coefficient between predicted responses $\hat{\mathbf{y}}_v$ and observed fMRI responses $\mathbf{y}_v$ for each vertex $v$:

$$r_v = \frac{\sum_{i=1}^{N_{\text{test}}} (\hat{y}_{i,v} - \bar{\hat{y}}_v)(y_{i,v} - \bar{y}_v)}{\sqrt{\sum_{i=1}^{N_{\text{test}}} (\hat{y}_{i,v} - \bar{\hat{y}}_v)^2 \cdot \sum_{i=1}^{N_{\text{test}}} (y_{i,v} - \bar{y}_v)^2}}$$

---

## 5. Anatomical Hierarchy & Statistical Metrics

### 1. Discrete Hierarchy Alignment (Spearman $\rho$)
Each visual area $R$ is assigned a canonical anatomical hierarchy rank $A_R \in \{1.0, 2.0, \dots, 6.0\}$ based on primate visual anatomy. For each model, the discrete peak layer depth $d_R^* = \arg\max_l r_{l,R}$ is identified. The discrete hierarchy score is computed as the Spearman rank correlation:

$$\rho_{\text{discrete}} = \text{SpearmanRank}\big(\{A_R\}, \{d_R^*\}\big)$$

### 2. Baseline-Subtracted Center-of-Mass Depth ($\bar{d}$)
In raw activations, deep convolutional layers often retain moderate predictivity in early visual cortex ($V1$) due to residual retinotopic information. Discrete argmax metrics can be sensitive to small fluctuations across adjacent layers.

To compute a continuous, robust measure of layer preference, the **baseline-subtracted Center-of-Mass depth** $\bar{d}_R$ weights each layer's normalized depth $d_l \in [0, 1]$ by its predictivity above the layer minimum:

$$\bar{d}_R = \frac{\sum_{l=1}^L d_l \cdot \max(0, r_{l,R} - \min_k r_{k,R})}{\sum_{l=1}^L \max(0, r_{l,R} - \min_k r_{k,R})}$$

$$\rho_{\text{CoM}} = \text{SpearmanRank}\big(\{A_R\}, \{\bar{d}_R\}\big)$$

### 3. Non-Parametric Significance Testing
* **Permutation Tests ($N=10,000$)**: The null hypothesis states that model layer depth has no relationship to anatomical cortical rank. Anatomical ranks $\{A_R\}$ are shuffled $10,000$ times to generate an empirical null distribution and compute two-tailed $p$-values.
* **Paired Image-Level Bootstrap ($N=1,000$)**: Test-set image predictions are resampled with replacement $1,000$ times to calculate 95% bootstrap confidence intervals for hierarchy correlations and predictivity differences.

### 4. The Noise Ceiling
Biological fMRI signals contain non-neural sources of variance, including scanner thermal noise, physiological fluctuations (respiration, cardiac cycles), and participant attentional shifts. The **split-half noise ceiling** estimates the theoretical upper bound of explainable variance. A model prediction of $r = 0.50$ in a brain region with an estimated noise ceiling of $r_{\text{ceil}} = 0.65$ explains approximately $77\%$ of the maximum possible predictable neural variance.

---

## 6. Key Scientific Takeaways

### 1. Hierarchy Preservation in Trained Vision Models
All task-trained neural networks exhibit statistically significant positive alignment with the cortical hierarchy ($\rho_{\text{discrete}} \ge +0.85$, $p < 0.01$; $\rho_{\text{CoM}} \ge +0.76$, $p < 0.02$). AlexNet achieves near-perfect monotonic alignment ($\rho_{\text{CoM}} = +0.985$, 95% CI $[0.90, 1.00]$).

### 2. Untrained Networks Invert Biological Hierarchy
The untrained ResNet-50 control exhibits a strongly negative rank correlation ($\rho_{\text{CoM}} = -0.966$, $p < 10^{-4}$). In untrained networks, deep layers predict early visual cortex ($V1$) better than shallow layers because repeated pooling downsamples spatial dimensions and aggregates random Gaussian noise. This confirms that **cortical hierarchy alignment is driven by visual learning**, rather than architectural inductive biases alone.

### 3. Ventral Stream Specialization in Multimodal CLIP and Deep ResNets
While early visual areas ($V1$) are best predicted by early convolutional layers ($r = 0.5726$), high-level category-selective areas ($FFA$, $PPA$, $RSC$) are predicted best by deep stages of CLIP ViT-B/16 and ResNet-50 ($r \approx 0.50 - 0.58$), reflecting the emergence of complex semantic and domain representations.

---

## 7. Glossary: Neuroscience & Machine Learning

| Term | Category | Definition |
|:---|:---:|:---|
| **BOLD Signal** | Neuroscience | Blood-Oxygen-Level-Dependent signal measured by fMRI, reflecting local changes in blood oxygenation correlated with neural metabolic activity. |
| **Voxel / Vertex** | Neuroimaging | A voxel is a 3D volumetric pixel. A vertex is a single node on the 2D triangular surface mesh representing the folded cortical sheet. |
| **Retinotopy** | Neuroscience | Spatial mapping where adjacent points on the retina project to adjacent anatomical locations in primary visual cortex ($V1$). |
| **Receptive Field** | Vision / ML | The region of visual input space that evokes a response in a sensory neuron or artificial convolutional filter. |
| **Ventral Visual Stream** | Neuroscience | The neural pathway extending from $V1$ through inferior temporal cortex, specialized for object identity, face recognition, and scene perception ("what" pathway). |
| **Category-Selective Area** | Neuroscience | A specialized cortical subregion responding preferentially to a specific visual category (e.g., $FFA$ for faces, $PPA$ for places, $EBA$ for bodies). |
| **Linear Encoding Model** | Computational Neuroscience | A regression model that tests whether neural responses can be linearly reconstructed from a computational feature space. |
| **Ridge Regression** | Machine Learning | Linear regression with an $L_2$ regularizer ($\alpha \|\mathbf{W}\|_2^2$) to stabilize weight estimates and prevent overfitting with correlated features. |
| **Spearman $\rho$** | Statistics | Non-parametric rank correlation measuring the monotonic relationship between two ranked variables. |
| **Center-of-Mass Depth** | Analysis | A continuous, weighted-average metric identifying the effective layer depth preferred by a given cortical region. |
| **Noise Ceiling** | Neuroimaging | The theoretical maximum prediction accuracy achievable given physiological and measurement noise in the biological data. |

---

## 8. Navigating the Repository

```
├── configs/               # Hyperparameters, model taps, PCA dimensions, alpha grids
├── data/                  # Data loaders, ROI parsers, COCO category label mappings
├── features/              # Feature extraction, spatial pooling, and feature caching
├── encoding/              # Voxel-wise Ridge regression engine and train-set PCA
├── analysis/              # Hierarchy statistics, bootstrap tests, and figure generation
├── app/                   # Standalone interactive web explorer dashboard
├── figures/               # Generated analysis figures (Figures 1–7)
├── notebooks/             # Self-contained Google Colab / Kaggle runner
├── RESULTS.md             # Quantitative results and exact layer-by-ROI tables
└── README.md              # Project landing page and reproduction guide
```

To run the pipeline and explore the results:
```bash
# Verify pipeline on synthetic data
python -m tests.test_pipeline

# Rebuild all tables and statistical tests from predictions
python analysis/rebuild_tables_and_stats.py

# Launch interactive web dashboard
python app/app.py
```
