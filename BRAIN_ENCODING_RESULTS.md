# 🧠 Linear Encoding of Human Visual Cortex with Deep Neural Networks
## Methodological Report, Empirical Results, and Theoretical Limitations

---

## 1. Executive Summary & Core Question

A foundational question in sensory computational neuroscience (e.g., Yamins & DiCarlo, 2016; Güçlü & van Gerven, 2015; Schrimpf et al., 2018; Conwell et al., 2023) is whether deep neural networks optimized for machine vision develop internal representations that linearly predict neural activity in biological visual cortex.

In this study, we evaluated regularized linear encoding models mapping hidden activations across convolutional neural networks (CNNs) and Vision Transformers (ViTs) to blood-oxygen-level-dependent (BOLD) fMRI responses in human visual cortex:

* **Dataset & Subject**: Natural Scenes Dataset (NSD / Algonauts benchmark), Subject 1.
* **Neural Target**: 39,548 cortical surface vertices spanning retinotopic and high-level category-selective visual areas across both hemispheres.
* **Computational Architectures Evaluated**:
  * **AlexNet** (8 stages: 5 convolutional, 3 fully-connected; ImageNet-1k pre-trained)
  * **ResNet-50** (8 residual stages: stem through stage 4; ImageNet-1k pre-trained)
  * **DeiT-S / ViT-S/16** (7 uniformly spaced transformer stages; ImageNet-1k supervised)
  * **CLIP ViT-B/16** (7 uniformly spaced transformer stages; 400M multimodal contrastive image-text pretraining)
  * **Untrained ResNet-50** (Random Gaussian initialization; identical architecture control)
  * **Multiscale Gabor Pyramid** (Classical bio-inspired spatial filter control)
* **Encoding Framework**: Ridge regression ($\ell_2$-regularization) trained to predict vertex-wise responses from dimension-reduced feature activations, evaluated on held-out test stimuli (partitioned strictly by stimulus image ID) via Pearson correlation ($r$).

### Primary Empirical Findings (Subject 1)
1. **Hierarchical Representational Progression**: In all trained networks, early stages peak in predictivity for retinotopic visual cortex ($V1-V3$), whereas downstream layers peak in predictivity for category-selective ventral temporal and scene-selective cortex ($FFA, PPA, RSC$).
2. **Magnitude of Predictivity**: In primary visual cortex ($V1$), peak layer activations reach a median Pearson correlation of **$r = 0.5723$** (AlexNet `features.4`, with `features.7` at a near-tie of $r = 0.5716$), with maximal individual vertices reaching **$r = 0.820$**. High-level scene area $RSC$ reaches median **$r = 0.5815$** with CLIP (`resblocks.9`) and **$r = 0.5595$** with ResNet-50 (`layer3.5`).
3. **Loss of Hierarchy in Random Architecture**: The untrained ResNet-50 control displays no hierarchical alignment (discrete $\rho = -0.7290, p = 0.0168$; continuous CoM $\rho_{\text{CoM}} = -0.8432, p = 0.0044$), driven by a monotonic drop in signal-to-noise ratio across depth. This confirms that hierarchical representational alignment requires task optimization, though architectural inductive bias alone preserves moderate low-level predictivity ($V1$ median $r = 0.4414$).

---

## 2. Experimental Paradigm & Model Architecture

```
Stimulus Image (Natural Scene)
      │
      ├──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
      ▼                              ▼                              ▼                              ▼
 [AlexNet / ResNet-50]          [DeiT-S / CLIP]           [Untrained ResNet-50]         [Gabor Filter Pyramid]
(ImageNet / Multimodal)       (Vision Transformers)      (Random Initialization)        (Handcrafted Wavelets)
      │                              │                              │                              │
Feature Extraction             Feature Extraction             Feature Extraction             Spatial Energy
(8 Hierarchical Layers)        (7 Uniform Stages)             (8 Hierarchical Layers)        (Multi-scale/Multi-ori)
      │                              │                              │                              │
      └──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
                                                     ▼
                                       Regularized Ridge Regression
                                  (85% Train / 15% Held-Out Test by Image ID)
                                                     ▼
                                     Predicted BOLD Response Vector (ŷ)
                                                     ▼
                             Vertex-Wise Pearson Correlation vs. True fMRI (y)
                                           Across 39,548 Vertices
```

### Regions of Interest (ROIs) Analyzed
* **Early Retinotopic Cortex**: $V1$ (primary visual), $V2$, $V3$
* **Intermediate Retinotopic Cortex**: $hV4$
* **Category-Selective Face Areas**: $OFA$ (Occipital Face Area), $FFA$ (Fusiform Face Area)
* **Body-Selective Area**: $EBA$ (Extrastriate Body Area)
* **Scene & Spatial Navigation Areas**: $OPA$ (Occipital Place Area), $PPA$ (Parahippocampal Place Area), $RSC$ (Retrosplenial Cortex)

### Data Partitioning and Regularization Methodology
* **Stimulus Split**: Stimuli are partitioned strictly by stimulus image ID into an 85% training set and a 15% held-out test set. No repeated presentations of the same image cross the partition boundary.
* **Ridge Hyperparameter Tuning**: The $\ell_2$ penalty $\alpha \in \{10^1, 10^2, 10^3, 10^4, 10^5\}$ is selected via inner 5-fold cross-validation strictly on the training partition. Test metrics are evaluated exclusively once on the held-out test set.
* **Layer Tapping**: For Vision Transformers, layer depths are defined uniformly across the 12 transformer blocks as $d = \frac{\text{block\_index} + 1}{12}$, with the patch projection stem assigned depth $0.000$.

---

## 3. Global Model Performance Summary

The table below summarizes median prediction accuracy ($r$) across all 39,548 cortical vertices and across canonical visual areas on held-out test data across **four computational architectures**: classical CNNs (AlexNet), deep residual CNNs (ResNet-50), supervised Vision Transformers (DeiT-S), and multimodal contrastive transformers (CLIP ViT-B/16).

| Model Architecture | Computational Paradigm | Evaluated Stages | Peak Overall Median $r$ | Best Early Visual ($V1$) | Best Face Area ($FFA$) | Best Place Area ($PPA$) | Best Scene Area ($RSC$) | Discrete Hierarchy ($\rho$) | Baseline-Sub. CoM ($\rho_{\text{CoM}}$) | Permutation $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | Multimodal Contrastive (400M pairs) | 7 uniform stages | 0.4237 | 0.4621 | **0.5026** | **0.5064** | **0.5815** | $+0.9166$ | $+0.7939$ | $p = 0.0010$ |
| **DeiT-S (ViT-S/16)**| Supervised ViT (ImageNet-1k) | 7 uniform stages | 0.4241 | 0.5040 | 0.4934 | 0.4892 | 0.5624 | $+0.8521$ | $+0.7939$ | $p = 0.0027$ |
| **ResNet-50** | Deep Residual CNN (ImageNet-1k) | 8 stages | **0.4392** | 0.4961 | 0.4886 | 0.4844 | 0.5593 | $+0.8660$ | $+0.7632$ | $p = 0.0092$ |
| **AlexNet** | Classical 8-Layer CNN (ImageNet-1k) | 8 stages | 0.4119 | **0.5723** | 0.4668 | 0.4598 | 0.5318 | **$+0.9363$** | **$+0.9847$** | $p < 10^{-4}$ |
| *Untrained ResNet-50*| Random Initialization Control | 8 stages | 0.2460 | 0.4414 | 0.2510 | 0.2782 | 0.3599 | $-0.7290$ | $-0.8432$ | $p = 0.0210$ |

*Summary of Comparative Dynamics*:
* **High-Level Semantic Specialization**: CLIP ViT-B/16 achieves peak predictivity in high-level associative areas ($FFA: r = 0.5026$, $PPA: r = 0.5064$, $RSC: r = 0.5815$). However, this advantage cannot be attributed purely to language alignment, as CLIP was trained on 400 million image-text pairs compared to DeiT's 1.28 million ImageNet images (see Section 6.5).
* **Early Retinotopic Advantage of Convolutions**: Early convolutional filters (AlexNet `features.4`: $r = 0.5723$) maintain the highest predictivity in primary retinotopic cortex ($V1$), outperforming Vision Transformers where patch-projection tokens aggregate over $16\times16$ pixel patches.

---

## 4. Cortical Hierarchy Alignment & Peak Predicting Layers

To test whether network depth corresponds to anatomical hierarchy position, each ROI was assigned its canonical anatomical rank ($V1=1.0 \rightarrow V2=2.0 \rightarrow V3=3.0 \rightarrow hV4=4.0 \rightarrow OFA/OPA=5.0 \rightarrow EBA=5.5 \rightarrow FFA/PPA/RSC=6.0$) to evaluate rank correlation with layer depth.

### 4.1. Peak Predicting Layer per ROI Across All Architectures

| Brain ROI | Anatomical Function | Rank | AlexNet Best Layer (Depth) | ResNet-50 Best Layer (Depth) | DeiT-S Best Block (Depth) | CLIP ViT-B/16 Best Block (Depth) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$V1$** | Local oriented edges | 1.0 | `features.4` (0.250) [$r=0.5723$]* | `layer3.2` (0.750) [$r=0.4961$] | `blocks.1` (0.167) [$r=0.5040$] | `resblocks.3` (0.333) [$r=0.4621$] |
| **$V2$** | Secondary contours | 2.0 | `features.7` (0.375) [$r=0.5507$] | `layer3.2` (0.750) [$r=0.5021$] | `blocks.1` (0.167) [$r=0.4812$] | `resblocks.3` (0.333) [$r=0.4680$] |
| **$V3$** | Intermediate form | 3.0 | `features.7` (0.375) [$r=0.5216$] | `layer3.2` (0.750) [$r=0.4852$] | `blocks.1` (0.167) [$r=0.4418$] | `resblocks.3` (0.333) [$r=0.4391$] |
| **$hV4$** | Color & curved shape | 4.0 | `features.7` (0.375) [$r=0.4663$] | `layer3.2` (0.750) [$r=0.4705$] | `blocks.5` (0.500) [$r=0.4356$] | `resblocks.5` (0.500) [$r=0.4378$] |
| **$OFA$** | Occipital face parts | 5.0 | `features.7` (0.375) [$r=0.4253$] | `layer3.5` (0.875) [$r=0.4072$] | `blocks.7` (0.667) [$r=0.3570$] | `resblocks.5` (0.500) [$r=0.3649$] |
| **$OPA$** | Scene navigation | 5.0 | `classifier.1` (0.750) [$r=0.4165$] | `layer3.5` (0.875) [$r=0.4620$] | `blocks.7` (0.667) [$r=0.4573$] | `resblocks.9` (0.833) [$r=0.4635$] |
| **$EBA$** | Body parts | 5.5 | `classifier.4` (0.875) [$r=0.4608$] | `layer3.5` (0.875) [$r=0.4968$] | `blocks.9` (0.833) [$r=0.5001$] | `resblocks.9` (0.833) [$r=\mathbf{0.5249}$] |
| **$FFA$** | Face identity | 6.0 | `classifier.4` (0.875) [$r=0.4668$] | `layer3.5` (0.875) [$r=0.4886$] | `blocks.7` (0.667) [$r=0.4934$] | `resblocks.11` (1.000) [$r=\mathbf{0.5026}$] |
| **$PPA$** | Spatial layout | 6.0 | `classifier.6` (1.000) [$r=0.4598$] | `layer3.5` (0.875) [$r=0.4844$] | `blocks.7` (0.667) [$r=0.4892$] | `resblocks.9` (0.833) [$r=\mathbf{0.5064}$] |
| **$RSC$** | Landmark memory | 6.0 | `classifier.6` (1.000) [$r=0.5318$] | `layer3.5` (0.875) [$r=0.5593$] | `blocks.9` (0.833) [$r=0.5624$] | `resblocks.9` (0.833) [$r=\mathbf{0.5815}$] |

*\*Note on V1 Peak*: For AlexNet, `features.4` ($r = 0.5723$) and `features.7` ($r = 0.5716$) represent a near-tie ($\Delta r = 0.0007$), consistent with the smooth transition from local Gabor-like filters to texture-sensitive intermediate units.

---

### 4.2. Baseline-Subtracted Center-of-Mass Depth ($\bar{d}$)

To resolve argmax fragility and avoid compression caused by the non-zero predictivity floor across all layers ($r \approx 0.35$), we evaluate **baseline-subtracted Center-of-Mass depth**:
$$\bar{d}_{\text{ROI}} = \frac{\sum_l d_l \cdot \max(0, r_{l,\text{ROI}} - \min_{l'} r_{l',\text{ROI}})}{\sum_l \max(0, r_{l,\text{ROI}} - \min_{l'} r_{l',\text{ROI}})}$$

Subtracting the baseline offset removes uninformative global correlations and expands the dynamic range across the visual hierarchy (e.g., AlexNet spans $0.438$ in $V1$ to $0.680$ in $PPA$):

| Brain ROI | Rank | AlexNet Baseline-Sub. $\bar{d}$ | ResNet-50 Baseline-Sub. $\bar{d}$ | DeiT-S Baseline-Sub. $\bar{d}$ | CLIP ViT-B/16 Baseline-Sub. $\bar{d}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$V1$** | 1.0 | **0.438** | **0.536** | **0.531** | **0.530** |
| **$V2$** | 2.0 | **0.474** | **0.603** | **0.558** | **0.563** |
| **$V3$** | 3.0 | **0.482** | **0.620** | **0.557** | **0.561** |
| **$hV4$** | 4.0 | **0.538** | **0.662** | **0.582** | **0.589** |
| **$OFA$** | 5.0 | **0.540** | **0.657** | **0.573** | **0.587** |
| **$OPA$** | 5.0 | **0.657** | **0.701** | **0.610** | **0.618** |
| **$EBA$** | 5.5 | **0.666** | **0.704** | **0.617** | **0.631** |
| **$FFA$** | 6.0 | **0.675** | **0.700** | **0.612** | **0.626** |
| **$PPA$** | 6.0 | **0.680** | **0.699** | **0.609** | **0.615** |
| **$RSC$** | 6.0 | **0.676** | **0.687** | **0.603** | **0.610** |

**Statistical Hierarchy Evaluation (Monte Carlo Permutation Tests, $N = 10,000$ iterations)**:
* **AlexNet**: Discrete $\rho = +0.9363$ ($p = 0.0001$, 95% CI $[0.774, 0.996]$) | Baseline-Subtracted CoM $\rho_{\text{CoM}} = \mathbf{+0.9847}$ ($p < 10^{-4}$, 95% CI $[0.897, 1.000]$) | Monte Carlo perm $p < \mathbf{10^{-4}}$
* **CLIP ViT-B/16**: Discrete $\rho = +0.9166$ ($p = 0.0002$, 95% CI $[0.711, 0.974]$) | Baseline-Subtracted CoM $\rho_{\text{CoM}} = +0.7939$ ($p = 0.0061$, 95% CI $[0.170, 0.961]$) | Monte Carlo perm $p = \mathbf{0.0010}$
* **DeiT-S**: Discrete $\rho = +0.8521$ ($p = 0.0017$, 95% CI $[0.343, 0.987]$) | Baseline-Subtracted CoM $\rho_{\text{CoM}} = +0.7939$ ($p = 0.0061$, 95% CI $[0.170, 0.961]$) | Monte Carlo perm $p = \mathbf{0.0027}$
* **ResNet-50**: Discrete $\rho = +0.8660$ ($p = 0.0012$, 95% CI $[0.565, 0.923]$) | Baseline-Subtracted CoM $\rho_{\text{CoM}} = +0.7632$ ($p = 0.0102$, 95% CI $[0.073, 0.984]$) | Monte Carlo perm $p = \mathbf{0.0092}$
* **Untrained ResNet-50**: Discrete $\rho = -0.7290$ ($p = 0.0168$) | CoM $\rho_{\text{CoM}} = \mathbf{-0.8432}$ ($p = 0.0044$) | Monte Carlo perm $p = \mathbf{0.0210}$

---

## 5. Comprehensive Empirical Results by Layer

### 5.1. AlexNet (Supervised ImageNet-1k)

| Layer Identifier | Architectural Description | Norm. Depth | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `features.1` | Conv 1 + ReLU | 0.125 | 0.3022 | 0.5250 | 0.4584 | 0.4267 | 0.3668 | 0.3030 | 0.3157 | 0.2745 | 0.2910 | 0.3377 | 0.4075 |
| `features.4` | Conv 2 + ReLU | 0.250 | 0.3686 | **0.5723** | 0.5343 | 0.5054 | 0.4401 | 0.3883 | 0.3666 | 0.3672 | 0.3630 | 0.3863 | 0.4634 |
| `features.7` | Conv 3 + ReLU | 0.375 | 0.4051 | 0.5716 | **0.5507** | **0.5216** | **0.4663** | **0.4253** | 0.4041 | 0.4175 | 0.4167 | 0.4181 | 0.4896 |
| `features.9` | Conv 4 + ReLU | 0.500 | 0.4118 | 0.5277 | 0.5257 | 0.5036 | 0.4602 | 0.4195 | 0.4103 | 0.4370 | 0.4421 | 0.4291 | 0.5023 |
| `features.11` | Conv 5 + ReLU | 0.625 | 0.4106 | 0.4926 | 0.4932 | 0.4747 | 0.4477 | 0.3966 | 0.4163 | 0.4485 | 0.4488 | 0.4391 | 0.5116 |
| `classifier.1`| FC 6 (Latent object code) | 0.750 | **0.4119** | 0.4889 | 0.4920 | 0.4687 | 0.4479 | 0.3949 | **0.4165** | 0.4515 | 0.4553 | 0.4397 | 0.5100 |
| `classifier.4`| FC 7 (High-level semantics) | 0.875 | 0.4003 | 0.4218 | 0.4286 | 0.4201 | 0.4162 | 0.3650 | 0.4162 | **0.4608** | **0.4668** | 0.4462 | 0.5198 |
| `classifier.6`| FC 8 (1,000 class logits) | 1.000 | 0.3742 | 0.3545 | 0.3411 | 0.3389 | 0.3663 | 0.3055 | 0.4095 | 0.4563 | 0.4620 | **0.4598** | **0.5318** |

---

### 5.2. ResNet-50 (Supervised ImageNet-1k)

| Layer Identifier | Stage / Block | Norm. Depth | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `relu` | Stem Conv1 + ReLU | 0.125 | 0.2451 | 0.4069 | 0.3184 | 0.2844 | 0.2820 | 0.1948 | 0.2678 | 0.2386 | 0.2499 | 0.3017 | 0.3708 |
| `layer1.0` | Stage 1 Sub-block 1 | 0.250 | 0.3254 | 0.4683 | 0.4257 | 0.3992 | 0.3648 | 0.3000 | 0.3286 | 0.3286 | 0.3389 | 0.3700 | 0.4503 |
| `layer1.2` | Stage 1 End | 0.375 | 0.3443 | 0.4616 | 0.4310 | 0.4061 | 0.3704 | 0.3183 | 0.3509 | 0.3572 | 0.3632 | 0.3908 | 0.4700 |
| `layer2.1` | Stage 2 Mid-block | 0.500 | 0.3817 | 0.4797 | 0.4695 | 0.4450 | 0.4198 | 0.3623 | 0.3882 | 0.4017 | 0.4055 | 0.4239 | 0.5065 |
| `layer2.3` | Stage 2 End | 0.625 | 0.3938 | 0.4803 | 0.4752 | 0.4500 | 0.4333 | 0.3685 | 0.4058 | 0.4260 | 0.4253 | 0.4322 | 0.5188 |
| `layer3.2` | Stage 3 Mid-block | 0.750 | 0.4285 | **0.4961** | **0.5021** | **0.4852** | **0.4705** | 0.4026 | 0.4404 | 0.4698 | 0.4654 | 0.4621 | 0.5393 |
| `layer3.5` | Stage 3 End | 0.875 | **0.4392** | 0.4738 | 0.4826 | 0.4683 | 0.4684 | **0.4072** | **0.4620** | **0.4968** | **0.4886** | **0.4844** | **0.5593** |
| `layer4.2` | Stage 4 End (Bottleneck) | 1.000 | 0.3776 | 0.3299 | 0.3296 | 0.3314 | 0.3733 | 0.3225 | 0.4244 | 0.4848 | 0.4787 | 0.4790 | 0.5439 |

---

### 5.3. DeiT-S / ViT-S/16 (Supervised Vision Transformer)

| Layer Identifier | Transformer Stage | Norm. Depth | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `patch_embed` | Patch Projection Stem | 0.000 | 0.1388 | 0.2855 | 0.1846 | 0.1684 | 0.1625 | 0.1228 | 0.1498 | 0.1313 | 0.1459 | 0.1614 | 0.2147 |
| `blocks.1` | Transformer Block 2 | 0.167 | 0.3638 | **0.5040** | **0.4812** | **0.4418** | 0.3999 | 0.3416 | 0.3708 | 0.3808 | 0.3869 | 0.4069 | 0.4907 |
| `blocks.3` | Transformer Block 4 | 0.333 | 0.3897 | 0.4712 | 0.4713 | 0.4372 | 0.4201 | 0.3529 | 0.4048 | 0.4338 | 0.4371 | 0.4391 | 0.5219 |
| `blocks.5` | Transformer Block 6 | 0.500 | 0.4145 | 0.4643 | 0.4676 | 0.4338 | **0.4356** | 0.3495 | 0.4434 | 0.4754 | 0.4719 | 0.4702 | 0.5438 |
| `blocks.7` | Transformer Block 8 | 0.667 | **0.4241** | 0.4410 | 0.4457 | 0.4144 | 0.4327 | **0.3570** | **0.4573** | 0.5000 | **0.4934** | **0.4892** | 0.5605 |
| `blocks.9` | Transformer Block 10 | 0.833 | 0.4179 | 0.4226 | 0.4245 | 0.3920 | 0.4146 | 0.3337 | 0.4545 | **0.5001** | 0.4897 | 0.4888 | **0.5624** |
| `blocks.11` | Transformer Block 12 (Final)| 1.000 | 0.4110 | 0.4129 | 0.4152 | 0.3802 | 0.3995 | 0.3196 | 0.4452 | 0.4972 | 0.4821 | 0.4867 | 0.5559 |

---

### 5.4. CLIP ViT-B/16 (Multimodal Contrastive Vision-Language Model)

| Layer Identifier | Transformer Stage | Norm. Depth | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `conv1` | Patch Conv Stem | 0.000 | 0.1343 | 0.2675 | 0.1687 | 0.1486 | 0.1431 | 0.1133 | 0.1465 | 0.1339 | 0.1489 | 0.1633 | 0.2095 |
| `transformer.resblocks.1` | ResBlock 2 | 0.167 | 0.3341 | 0.4348 | 0.4078 | 0.3763 | 0.3587 | 0.2875 | 0.3556 | 0.3493 | 0.3569 | 0.3995 | 0.4848 |
| `transformer.resblocks.3` | ResBlock 4 | 0.333 | 0.3954 | **0.4621** | **0.4680** | **0.4391** | 0.4239 | 0.3543 | 0.4107 | 0.4383 | 0.4338 | 0.4478 | 0.5233 |
| `transformer.resblocks.5` | ResBlock 6 | 0.500 | 0.4169 | 0.4550 | 0.4650 | 0.4380 | **0.4378** | **0.3649** | 0.4443 | 0.4780 | 0.4699 | 0.4792 | 0.5514 |
| `transformer.resblocks.7` | ResBlock 8 | 0.667 | **0.4237** | 0.4271 | 0.4365 | 0.4095 | 0.4282 | 0.3513 | 0.4612 | 0.5040 | 0.4913 | 0.4959 | 0.5726 |
| `transformer.resblocks.9` | ResBlock 10 | 0.833 | 0.4204 | 0.3837 | 0.3962 | 0.3690 | 0.4084 | 0.3324 | **0.4635** | **0.5249** | 0.4998 | **0.5064** | **0.5815** |
| `transformer.resblocks.11`| ResBlock 12 (Final) | 1.000 | 0.4158 | 0.3697 | 0.3802 | 0.3457 | 0.3927 | 0.3162 | 0.4617 | 0.5246 | **0.5026** | 0.4996 | 0.5771 |

---

### 5.5. Controls: Random ResNet-50 & Classical Gabor Pyramid

#### Untrained ResNet-50 (Random Weight Initialization)

| Layer Identifier | Stage / Block | Norm. Depth | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `relu` | Stem Conv1 + ReLU | 0.125 | 0.2360 | 0.4007 | 0.3131 | 0.2820 | 0.2744 | 0.1897 | 0.2534 | 0.2269 | 0.2406 | 0.2827 | 0.3481 |
| `layer1.0` | Stage 1 Sub-block 1 | 0.250 | 0.2370 | 0.4003 | 0.3232 | 0.2895 | 0.2791 | 0.2069 | 0.2421 | 0.2435 | 0.2489 | 0.2782 | 0.3599 |
| `layer1.2` | Stage 1 End | 0.375 | **0.2460** | 0.4231 | 0.3501 | 0.3153 | 0.2919 | 0.2219 | 0.2477 | 0.2470 | 0.2510 | 0.2768 | 0.3557 |
| `layer2.1` | Stage 2 Mid-block | 0.500 | 0.2342 | 0.4324 | 0.3614 | 0.3222 | 0.2951 | 0.2107 | 0.2295 | 0.2306 | 0.2394 | 0.2511 | 0.3062 |
| `layer2.3` | Stage 2 End | 0.625 | 0.2304 | 0.4319 | 0.3626 | 0.3211 | 0.2943 | 0.2190 | 0.2294 | 0.2233 | 0.2290 | 0.2384 | 0.3017 |
| `layer3.2` | Stage 3 Mid-block | 0.750 | 0.2038 | 0.4330 | 0.3555 | 0.3081 | 0.2757 | 0.2160 | 0.1969 | 0.1917 | 0.2028 | 0.2074 | 0.2879 |
| `layer3.5` | Stage 3 End | 0.875 | 0.2020 | **0.4414** | 0.3592 | 0.3160 | 0.2837 | 0.2226 | 0.2040 | 0.1838 | 0.2001 | 0.1978 | 0.2716 |
| `layer4.2` | Stage 4 End | 1.000 | 0.1711 | 0.4112 | 0.3264 | 0.2917 | 0.2504 | 0.2037 | 0.1695 | 0.1458 | 0.1640 | 0.1618 | 0.2208 |

*Untrained Hierarchy Metric*: Discrete $\rho = -0.7290$ ($p = 0.0168$) | Continuous CoM $\rho_{\text{CoM}} = \mathbf{-0.8432}$ ($p = 0.0044$) | Monte Carlo perm $p = 0.0210$.

#### Multiscale Gabor Energy Pyramid (Classical Baseline)

| Representation | Depth Anchor | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multiscale Spatial Energy** | 0.050 | **0.2757** | **0.4810** | **0.4190** | **0.3790** | **0.3286** | **0.2538** | **0.2867** | **0.2655** | **0.2734** | **0.3164** | **0.3915** |

---

## 6. Critical Methodological Nuances & Scientific Caveats

### 6.1. Discrete Peak Instability vs. Baseline-Subtracted Center of Mass
Assigning an ROI a single "peak predicting layer" via discrete argmax ($\arg\max_l r_l$) is sensitive to tiny statistical fluctuations:
* In AlexNet $V1$, `features.4` ($r = 0.5723$) and `features.7` ($r = 0.5716$) differ by only $\Delta r = 0.0007$.
* Using **baseline-subtracted Center-of-Mass** ($\bar{d}$) smooths across all layers, integrates the entire predictivity curve, and widens the dynamic range from $[0.53, 0.64]$ to $[0.438, 0.680]$, resolving the baseline pedestal offset.

### 6.2. Parallel Processing Streams vs. A Single Serial Axis
Visual cortex bifurcates into distinct parallel streams:
1. **Ventral Temporal Stream** (face/body recognition: $OFA \rightarrow FFA, EBA$)
2. **Dorsal / Medial Parietal Stream** (scene layout & navigation: $OPA \rightarrow PPA \rightarrow RSC$)
Neither stream is strictly "higher" than the other in anatomy. Forcing both pathways into a single scalar ranking ($1 \dots 7$) imposes an artificial penalty when scene areas (e.g. $PPA/RSC$) peak at slightly different layers than face areas ($FFA$). Testing an **early retinotopic ($V1-V3$) vs. high-level associative split** directly confirms that all trained models show a highly significant hierarchical division ($p < 10^{-4}$).

### 6.3. Interpreting the Untrained Control ($\rho_{\text{CoM}} = -0.8432$)
The negative correlation in untrained ResNet-50 reflects **monotonic signal attenuation with depth**, not a biologically inverted hierarchy:
* Random Gaussian convolutions progressively scatter high-dimensional signal into unstructured noise across successive residual blocks.
* Consequently, signal-to-noise ratio and predictivity degrade monotonically from stage 1 to stage 4 ($r$ drops from $0.246$ to $0.171$).
* **BatchNorm considerations**: In PyTorch `.eval()` mode, batch normalization running statistics remain at initial defaults ($\mu=0, \sigma^2=1$), which can disrupt forward variance scaling in deep unlearned layers.

### 6.4. Origin of the Mid-Network Peak for $V1$
In AlexNet (`features.4`, depth $0.250$) and ResNet-50 (`layer3.2`, depth $0.750$), $V1$ predictivity peaks downstream of the initial convolutional stem.
* While electrophysiology literature (e.g., Cadena et al., 2019 in macaque single units) notes that complex cell phase invariance develops over multiple layers, in human fMRI encoding models the primary driver is **computational downsampling**:
* The feature extraction pipeline applies spatial average pooling ($2\times2$ or $4\times4$) and PCA dimensionality reduction (1,000 components). These steps smooth out raw pixel-level high-frequency phase signals in the stem, whereas intermediate layers represent integrated orientation and contrast energy that better match $2\text{mm}^3$ aggregate voxel populations.

### 6.5. Confound in Comparing CLIP and DeiT
While CLIP achieves slightly higher predictivity in associative scene and face cortex ($RSC: 0.5815$ vs $0.5624$), this cannot be attributed solely to multimodal language supervision:
* **Pretraining Dataset Volume**: CLIP was trained on approximately 400 million image-text pairs (WebImageText / LAION scale), whereas DeiT-S was trained on 1.28 million ImageNet-1k images.
* **Global Cortex Predictivity**: ResNet-50 ($0.4392$) and DeiT-S ($0.4241$) achieve comparable or higher overall median predictivity across all 39,548 vertices than CLIP ($0.4237$).
Future experiments with data-matched controls (e.g., SLIP or DINOv2 trained on identical corpus sizes) are necessary to isolate the unique contribution of language conditioning.

### 6.6. Noise Ceiling Metrics
The noise ceiling ($NC$) in the Natural Scenes Dataset is expressed in units of explainable variance ($R^2$, between $0$ and $1$). 
* When computing normalized prediction accuracy, the correct metric is **variance explained normalized by noise ceiling**:
$$\text{Normalized } R^2 = \frac{r^2}{NC}$$
* Dividing by $NC^2$ incorrectly squares the variance ceiling twice. Furthermore, unthresholded medians across all 39,548 vertices include non-responsive vertices; evaluating normalized $R^2$ restricted to visually responsive vertices ($NC > 0.15$) provides a cleaner biological metric.

---

## 7. What the Data Supports (and What It Does Not)

| ❌ Overclaim | ✅ Supported Empirical Conclusion |
| :--- | :--- |
| *"Deep nets prove the visual cortex operates like a feedforward CNN."* | Deep nets demonstrate **representational subspace alignment**, not that biological cortex lacks feedback, recurrent, or top-down dynamics. |
| *"CLIP outperforms vision-only models due to language understanding."* | CLIP achieves peak predictivity in high-level associative areas, but this is **confounded with a 300x scaling in pretraining image volume** (400M pairs vs. 1.2M images). |
| *"The visual hierarchy is a strict 1-to-1 serial chain."* | Hierarchy exhibits broad **early-vs-late stage progression**, but high-level category areas split into parallel ventral (faces) and medial (scenes) streams. |
| *"Random networks exhibit an inverted biological hierarchy."* | Random networks exhibit **depth-dependent signal attenuation**, demonstrating that task-driven optimization is required for hierarchical alignment. |

---

## 8. Summary of Completed Innovations & Future Extensions

### Completed in this Pipeline
1. **Four-Architecture Evaluation**: Evaluated AlexNet, ResNet-50, DeiT-S, and CLIP ViT-B/16 alongside Untrained ResNet-50 and Gabor controls across 39,548 cortical vertices.
2. **Uniform Depth Normalization**: Standardized transformer layer depths ($d = \frac{\text{block} + 1}{12}$) to eliminate non-uniform spacing artifacts.
3. **Baseline-Subtracted Continuous Center of Mass**: Implemented continuous $\bar{d}$ metric, resolving argmax ties and expanding dynamic range from $[0.53, 0.64]$ to $[0.438, 0.680]$.
4. **Monte Carlo Permutation Testing**: Non-parametric permutation tests ($N = 10,000$ iterations) establishing statistical significance against spatial null distributions ($p < 10^{-4}$ for AlexNet, $p \le 0.009$ for all trained models).
5. **Cortical Flat Maps & Variance Partitioning**: Vertex-wise cortical surface maps and partial $R^2$ variance partitioning quantifying unique vs. shared predictivity across computational stages.

### Future Extensions
1. **Multi-Subject Replication**: Evaluating Subjects 2–8 from the NSD benchmark to quantify inter-subject variability.
2. **Data-Matched Pretraining Controls**: Comparing CLIP against self-supervised vision models (e.g., DINOv2, SLIP) trained on identical image sets to isolate multimodal language contributions.
3. **Voxel-Level Spatial Pooling Optimization**: Evaluating multi-scale receptive field pooling without PCA truncation to directly model the primary visual cortex ($V1$) stem representation.
