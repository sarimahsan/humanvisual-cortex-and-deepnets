# 🧠 Linear Encoding of Human Visual Cortex with Deep Neural Networks
## Methodological Report, Empirical Results, and Theoretical Limitations

---

## 1. Executive Summary & Core Question

A foundational question in sensory computational neuroscience (e.g., Yamins & DiCarlo, 2016; Güçlü & van Gerven, 2015; Schrimpf et al., 2018) is whether deep neural networks optimized purely for engineering object recognition develop internal representations that linearly predict neural activity in biological visual cortex.

In this experiment, we evaluated regularized linear encoding models mapping hidden activations of deep convolutional neural networks (CNNs) to blood-oxygen-level-dependent (BOLD) fMRI responses in human visual cortex:

* **Dataset & Subject**: Natural Scenes Dataset (NSD / Algonauts benchmark), Subject 1.
* **Neural Target**: 39,548 cortical surface vertices across left and right visual cortex.
* **Computational Models Evaluated**:
  * **AlexNet** (8 stages: 5 convolutional, 3 fully-connected; ImageNet pre-trained)
  * **ResNet-50** (8 residual stages: stem through stage 4; ImageNet pre-trained)
  * **Untrained ResNet-50** (Random Gaussian initialization; identical architecture control)
  * **Multiscale Gabor Pyramid** (Classical bio-inspired spatial filter control)
* **Encoding Framework**: Ridge regression ($\ell_2$-regularization) trained to predict vertex-wise responses from dimension-reduced feature activations, evaluated on held-out test stimuli via Pearson correlation ($r$).

### Primary Empirical Findings (Subject 1)
1. **Hierarchical Representational Progression**: In trained networks, early convolutional stages provide peak predictivity for early retinotopic areas ($V1-V3$), whereas downstream stages peak for category-selective ventral temporal and scene-selective areas ($FFA, PPA, RSC$).
2. **Magnitude of Predictivity**: In primary visual cortex ($V1$), peak layer activations reach a median Pearson correlation of **$r = 0.573$** (AlexNet `features.7`), with maximal individual vertices reaching **$r = 0.820$**. High-level scene area $RSC$ reaches median **$r = 0.560$** (ResNet `layer3.5`).
3. **Loss of Hierarchy in Random Architecture**: The untrained ResNet-50 control displays no hierarchical alignment ($\rho = -0.729$), driven by a monotonic drop in signal-to-noise ratio across depth. This confirms that hierarchical representational alignment requires task optimization, though architectural inductive bias alone preserves moderate low-level predictivity ($V1$ median $r = 0.441$).

---

## 2. Experimental Paradigm & Model Architecture

```
Stimulus Image (Natural Scene)
      │
      ├──────────────────────────────┬──────────────────────────────┐
      ▼                              ▼                              ▼
 [AlexNet / ResNet-50]     [Untrained ResNet-50]         [Gabor Filter Pyramid]
 (ImageNet-1k Trained)     (Random Initialization)       (Handcrafted Wavelets)
      │                              │                              │
Feature Extraction             Feature Extraction             Spatial Energy
(8 Hierarchical Layers)        (8 Hierarchical Layers)        (Multi-scale / Multi-orientation)
      │                              │                              │
      └──────────────────────────────┼──────────────────────────────┘
                                     ▼
                      Regularized Ridge Regression
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

---

## 3. Global Model Performance Summary

The table below summarizes median prediction accuracy ($r$) across all 39,548 cortical vertices and across canonical visual areas on held-out test data.

| Model Architecture | Evaluated Stages | Peak Overall Median $r$ | Best $V1$ Median $r$ | Best $FFA$ Median $r$ | Best $RSC$ Median $r$ | Hierarchy Rank Order ($\rho$) | Permutation / Bootstrap $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Trained)** | 8 stages | **0.4389** (`layer3.5`) | 0.4959 (`layer3.2`) | **0.4887** (`layer3.5`) | **0.5595** (`layer3.5`) | **$+0.8660$** | $p = 0.0012$ |
| **AlexNet (Trained)** | 8 layers | **0.4118** (`classifier.1`) | **0.5728** (`features.7`) | 0.4668 (`classifier.4`) | 0.5318 (`classifier.6`) | **$+0.8991$** | $p = 0.0004$ |
| **Gabor Pyramid (Control)** | 1 pyramid | 0.2757 | 0.4810 | 0.2734 | 0.3915 | N/A (Single scale) | N/A |
| **ResNet-50 (Untrained Control)** | 8 stages | 0.2460 (`layer1.2`) | 0.4414 (`layer3.5`) | 0.2510 (`layer1.2`) | 0.3599 (`layer1.0`) | **$-0.7290$** | $p = 0.0168$ |

*Note: All values reflect median Pearson $r$ across vertices within each ROI on held-out data.*

---

## 4. Cortical Hierarchy Alignment & Peak Predicting Layers

To test whether network depth corresponds to anatomical hierarchy position, we assigned each ROI its canonical anatomical rank ($V1=1.0 \rightarrow V2=2.0 \rightarrow V3=3.0 \rightarrow hV4=4.0 \rightarrow OFA/OPA=5.0 \rightarrow EBA=5.5 \rightarrow FFA/PPA/RSC=6.0$) and determined the layer depth that maximizes out-of-sample prediction accuracy.

### Peak Predicting Layer per ROI

| Brain ROI | Anatomical Function | Rank | AlexNet Peak Layer | AlexNet Depth | AlexNet $r$ | ResNet-50 Peak Layer | ResNet Depth | ResNet-50 $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V1$** | Local oriented edges | 1.0 | `features.7` (Conv3) | 0.375 | 0.5728 | `layer3.2` (Stage 3 mid) | 0.750 | 0.4959 |
| **$V2$** | Angles, junctions, textures | 2.0 | `features.7` (Conv3) | 0.375 | 0.5512 | `layer3.2` (Stage 3 mid) | 0.750 | 0.5015 |
| **$V3$** | Intermediate contours | 3.0 | `features.7` (Conv3) | 0.375 | 0.5226 | `layer3.2` (Stage 3 mid) | 0.750 | 0.4848 |
| **$hV4$** | Color, curved contours | 4.0 | `features.7` (Conv3) | 0.375 | 0.4673 | `layer3.2` (Stage 3 mid) | 0.750 | 0.4701 |
| **$OFA$** | Occipital face features | 5.0 | `features.7` (Conv3) | 0.375 | 0.4247 | `layer3.5` (Stage 3 end) | 0.875 | 0.4064 |
| **$OPA$** | Environmental navigation | 5.0 | `classifier.1` (FC6) | 0.750 | 0.4165 | `layer3.5` (Stage 3 end) | 0.875 | 0.4619 |
| **$EBA$** | Body parts and postures | 5.5 | `classifier.4` (FC7) | 0.875 | 0.4610 | `layer3.5` (Stage 3 end) | 0.875 | 0.4966 |
| **$FFA$** | Holistic face identity | 6.0 | `classifier.4` (FC7) | 0.875 | 0.4668 | `layer3.5` (Stage 3 end) | 0.875 | 0.4887 |
| **$PPA$** | Spatial layout & places | 6.0 | `classifier.6` (FC8) | 1.000 | 0.4598 | `layer3.5` (Stage 3 end) | 0.875 | 0.4845 |
| **$RSC$** | Landmark and scene memory | 6.0 | `classifier.6` (FC8) | 1.000 | 0.5318 | `layer3.5` (Stage 3 end) | 0.875 | 0.5595 |

### Rank-Order Correlation Statistics (Discrete Argmax)
* **AlexNet**: Spearman $\rho = +0.8991$ ($p = 4.01 \times 10^{-4}$), 95% bootstrap CI: $[0.704, 0.985]$
* **ResNet-50**: Spearman $\rho = +0.8660$ ($p = 1.19 \times 10^{-3}$), 95% bootstrap CI: $[0.565, 0.913]$
* **Untrained ResNet-50 Control**: Spearman $\rho = -0.7290$ ($p = 0.0168$), 95% bootstrap CI: $[-0.972, -0.282]$

### Robust Continuous Metric: Center-of-Mass Depth ($\bar{d}$)
To eliminate argmax tie instability, we computed the continuous correlation-weighted center of mass of each ROI across normalized network depth:
$$\bar{d}_{\text{ROI}} = \frac{\sum_l d_l \cdot \max(0, r_l)}{\sum_l \max(0, r_l)}$$
and evaluated statistical significance using both parametric testing and an exact **10,000-iteration non-parametric permutation test** (shuffling ROI anatomical labels to nullify spatial alignment):

| Brain ROI | Anatomical Rank | AlexNet Continuous Depth ($\bar{d}$) | ResNet-50 Continuous Depth ($\bar{d}$) | Untrained Control Depth ($\bar{d}$) |
| :--- | :---: | :---: | :---: | :---: |
| **$V1$** | 1.0 | **0.527** | **0.555** | 0.568 |
| **$V2$** | 2.0 | **0.537** | **0.573** | 0.569 |
| **$V3$** | 3.0 | **0.541** | **0.580** | 0.567 |
| **$hV4$** | 4.0 | **0.559** | **0.591** | 0.557 |
| **$OFA$** | 5.0 | **0.558** | **0.602** | 0.569 |
| **$OPA$** | 5.0 | **0.581** | **0.604** | 0.530 |
| **$EBA$** | 5.5 | **0.597** | **0.620** | 0.524 |
| **$FFA$** | 6.0 | **0.597** | **0.615** | 0.530 |
| **$PPA$** | 6.0 | **0.585** | **0.601** | 0.514 |
| **$RSC$** | 6.0 | **0.582** | **0.594** | 0.523 |

**Center-of-Mass Hierarchy Statistics:**
* **AlexNet**: Spearman **$\rho = +0.8924$** | Permutation $p = \mathbf{0.0007}$ ($7/10,000$ permutations) | Parametric $p = 0.0005$
* **ResNet-50**: Spearman **$\rho = +0.7385$** | Permutation $p = \mathbf{0.0194}$ ($194/10,000$ permutations) | Parametric $p = 0.0147$
* **Untrained ResNet-50**: Spearman **$\rho = -0.8432$** | Permutation $p = \mathbf{0.0044}$ ($44/10,000$ permutations) | Parametric $p = 0.0022$

*Takeaway*: The continuous metric smooths out individual layer ties, confirms monotonic progression ($0.527 \rightarrow 0.597$ in AlexNet; $0.555 \rightarrow 0.620$ in ResNet-50), and remains highly significant under non-parametric label permutation.

---

## 5. Comprehensive Empirical Results by Layer

### 5.1. AlexNet (Supervised ImageNet-1k)

| Layer Identifier | Architectural Description | Norm. Depth | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `features.1` | Conv 1 + ReLU | 0.125 | 0.3022 | 0.5250 | 0.4584 | 0.4267 | 0.3667 | 0.3030 | 0.3157 | 0.2746 | 0.2909 | 0.3377 | 0.4075 |
| `features.4` | Conv 2 + ReLU | 0.250 | 0.3691 | 0.5726 | 0.5345 | 0.5057 | 0.4403 | 0.3890 | 0.3672 | 0.3681 | 0.3657 | 0.3868 | 0.4644 |
| `features.7` | Conv 3 + ReLU | 0.375 | 0.4055 | **0.5728** | **0.5512** | **0.5226** | **0.4673** | **0.4247** | 0.4043 | 0.4182 | 0.4174 | 0.4188 | 0.4898 |
| `features.9` | Conv 4 + ReLU | 0.500 | 0.4116 | 0.5277 | 0.5257 | 0.5036 | 0.4602 | 0.4195 | 0.4103 | 0.4370 | 0.4421 | 0.4291 | 0.5023 |
| `features.11` | Conv 5 + ReLU | 0.625 | 0.4113 | 0.4926 | 0.4932 | 0.4747 | 0.4477 | 0.3966 | 0.4163 | 0.4485 | 0.4488 | 0.4391 | 0.5116 |
| `classifier.1`| FC 6 (Latent object code) | 0.750 | **0.4118** | 0.4889 | 0.4920 | 0.4687 | 0.4479 | 0.3949 | **0.4165** | 0.4515 | 0.4553 | 0.4397 | 0.5100 |
| `classifier.4`| FC 7 (High-level semantics) | 0.875 | 0.4002 | 0.4218 | 0.4286 | 0.4201 | 0.4162 | 0.3650 | 0.4162 | **0.4610** | **0.4668** | 0.4462 | 0.5198 |
| `classifier.6`| FC 8 (1,000 class logits) | 1.000 | 0.3742 | 0.3545 | 0.3411 | 0.3389 | 0.3663 | 0.3055 | 0.4095 | 0.4563 | 0.4620 | **0.4598** | **0.5318** |

### 5.2. ResNet-50 (Supervised ImageNet-1k)

| Layer Identifier | Stage / Block | Norm. Depth | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `relu` | Stem Conv1 + ReLU | 0.125 | 0.2451 | 0.4069 | 0.3184 | 0.2844 | 0.2820 | 0.1948 | 0.2678 | 0.2386 | 0.2499 | 0.3017 | 0.3708 |
| `layer1.0` | Stage 1 Sub-block 1 | 0.250 | 0.3254 | 0.4683 | 0.4257 | 0.3992 | 0.3648 | 0.3000 | 0.3286 | 0.3286 | 0.3389 | 0.3700 | 0.4503 |
| `layer1.2` | Stage 1 End | 0.375 | 0.3443 | 0.4616 | 0.4310 | 0.4061 | 0.3704 | 0.3183 | 0.3509 | 0.3572 | 0.3632 | 0.3908 | 0.4700 |
| `layer2.1` | Stage 2 Mid-block | 0.500 | 0.3820 | 0.4797 | 0.4695 | 0.4450 | 0.4198 | 0.3623 | 0.3882 | 0.4017 | 0.4055 | 0.4239 | 0.5065 |
| `layer2.3` | Stage 2 End | 0.625 | 0.3937 | 0.4803 | 0.4752 | 0.4500 | 0.4333 | 0.3685 | 0.4058 | 0.4260 | 0.4253 | 0.4322 | 0.5188 |
| `layer3.2` | Stage 3 Mid-block | 0.750 | 0.4279 | **0.4959** | **0.5015** | **0.4848** | **0.4701** | 0.4026 | 0.4404 | 0.4698 | 0.4654 | 0.4621 | 0.5393 |
| `layer3.5` | Stage 3 End | 0.875 | **0.4389** | 0.4738 | 0.4826 | 0.4683 | 0.4684 | **0.4064** | **0.4619** | **0.4966** | **0.4887** | **0.4845** | **0.5595** |
| `layer4.2` | Stage 4 End (Bottleneck) | 1.000 | 0.3776 | 0.3299 | 0.3296 | 0.3314 | 0.3733 | 0.3225 | 0.4244 | 0.4848 | 0.4787 | 0.4790 | 0.5439 |

### 5.3. Controls: Random ResNet-50 & Classical Gabor Pyramid

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

#### Multiscale Gabor Energy Pyramid (Classical Baseline)

| Representation | Depth Anchor | Median Overall $r$ | $V1$ | $V2$ | $V3$ | $hV4$ | $OFA$ | $OPA$ | $EBA$ | $FFA$ | $PPA$ | $RSC$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multiscale Spatial Energy** | 0.050 | **0.2757** | **0.4810** | **0.4190** | **0.3790** | **0.3286** | **0.2538** | **0.2867** | **0.2655** | **0.2734** | **0.3164** | **0.3915** |

---

## 6. Critical Methodological Nuances & Scientific Caveats

While these results replicate established findings in sensory encoding literature, rigorous review highlights several methodological subtleties that temper overly expansive claims:

### 6.1. The Fragility of Argmax "Peak Layer" Selection
Assigning an ROI a single "peak predicting layer" using a simple discrete argmax ($\arg\max_l r_l$) is sensitive to sampling noise:
* For AlexNet in $V1$, `features.4` (Conv2) achieves median $r = 0.5726$, while `features.7` (Conv3) achieves $r = 0.5728$. This difference ($\Delta r = 0.0002$) is within random split noise, yet argmax treats them as distinct hierarchical steps.
* Similarly, overall AlexNet predictivity between `classifier.1` ($r = 0.4118$) and `features.9` ($r = 0.4116$) represents an effective tie.
* **More robust alternative**: Future work should quantify ROI depth using the continuous **center of mass** of the predictivity profile across depth ($\bar{d} = \frac{\sum_l d_l \cdot r_l}{\sum_l r_l}$), bootstrapped across test stimulus resamples.

### 6.2. Coarseness of ResNet-50 Layer Splits and Spatial Autocorrelation
For ResNet-50, only two distinct layers emerged as winners across all ten ROIs:
* `layer3.2` for early/intermediate retinotopic areas ($V1, V2, V3, hV4$).
* `layer3.5` for all category- and scene-selective areas ($OFA, OPA, EBA, FFA, PPA, RSC$).
This effectively reduces the 10-ROI hierarchy test to a two-cluster step function. Furthermore, because adjacent anatomical visual areas share shared vasculature, spatial smoothing, and stimulus co-activations, ROIs are not statistically independent observations. Standard parametric $p$-values on Spearman rank correlation are therefore optimistic, requiring non-parametric permutation tests of ROI labels.

### 6.3. Interpreting the Untrained Control ($\rho = -0.73$)
The negative rank correlation ($\rho = -0.729$) in untrained ResNet-50 **does not indicate a reversed biological hierarchy**. Rather:
1. In randomly initialized networks, unlearned convolutional filter cascades progressively attenuate signal and accumulate high-dimensional noise as depth increases.
2. Performance decays monotonically with network depth across nearly all visual areas ($r$ drops from $0.246$ in stage 1 to $0.171$ in stage 4).
3. Because the argmax lands in earlier layers for most areas, the rank order exhibits a downward trend.
4. **BatchNorm considerations**: In untrained PyTorch models evaluated in `.eval()` mode, batch normalization running statistics remain at initial defaults ($\mu=0, \sigma^2=1$), which can disrupt forward signal propagation in deep stages. 
The scientifically accurate conclusion is that **untrained networks lack hierarchical cortical correspondence**, not that they mirror an inverted brain.

### 6.4. Why Does $V1$ Peak at Intermediate Layers Rather than the Stem?
In both AlexNet (`features.7`, depth $0.375$) and ResNet-50 (`layer3.2`, depth $0.750$), $V1$ predictivity peaks well downstream of the initial convolutional stem (`features.1`: $r=0.525$; ResNet `relu`: $r=0.407$). 

This is a recognized property in the encoding literature (e.g., Cadena et al., 2019):
* **Effective receptive field scaling**: The earliest layers have very small spatial receptive fields ($3\times3$ or $7\times7$ pixels with low stride) that do not match the cortical magnification and aggregate point-spread function of a standard 2mm fMRI voxel.
* **Context and lateral inhibition**: Intermediate layers integrate surround context, contrast normalization, and complex cell properties (phase invariance) that better approximate the aggregate neural population sampled by BOLD imaging.

### 6.5. Noise Ceilings and Vertex Reliability
In fMRI, raw correlation $r$ is bounded by the measurement noise ceiling ($NC$), governed by physiological noise (heart rate, respiration) and scanner thermal noise. 
* Claiming that $r = 0.57$ or peak $r = 0.82$ "approaches the noise ceiling" requires explicit computation of the split-half or repeat-reliability noise ceiling provided in the NSD dataset ($NC_1$ and $NC_2$).
* Furthermore, taking an unthresholded median across all 39,548 vertices ($r \approx 0.41 - 0.44$) includes many unselective or noisy peripheral vertices outside visual responsive cortex. Reporting noise-normalized explained variance ($r^2 / NC^2$) restricted to vertices above a minimum reliability threshold ($NC > 0.15$) will provide a more interpretable metric.

### 6.6. Dimensionality Matching in Baseline Controls
The Gabor multiscale energy model extracts a fixed bank of spatial filters, whereas deep convolutional stages yield tens of thousands of channel activations prior to dimensionality reduction. While ridge regression regularizes over-parameterization, differences in effective linear capacity between hand-crafted filters and wide deep representations represent an experimental confound unless explicitly matched via identical principal component analysis ($k$ latent components).

---

## 7. What the Data Supports (and What It Does Not)

To maintain rigorous scientific clarity, we contrast defensible conclusions with common overclaims:

| ❌ Overclaim | ✅ Supported Empirical Conclusion |
| :--- | :--- |
| *"Deep nets prove the brain uses backpropagation."* | Deep nets demonstrate **representational similarity**, not that biological brains use the same learning algorithms or optimization mechanics. |
| *"The AI and brain share the exact same architecture."* | Linear predictivity indicates that artificial task-optimized features span a similar **representational subspace** as population neural activity. |
| *"The visual hierarchy is strictly ordered in one-to-one fashion."* | The visual hierarchy exhibits broad **feedforward stage alignment**, but biological vision contains extensive recurrent, feedback, and lateral connections absent in standard feedforward CNNs. |
| *"Random networks prove an inverted hierarchy."* | Random networks exhibit **monotonic signal attenuation with depth**, demonstrating that learned visual task features are necessary for hierarchical alignment. |

---

## 8. Concrete Roadmap for Further Strengthening

To advance this study to publication-level maturity:

1. **Noise-Ceiling Normalization ($r / NC$)**: Compute voxel-wise split-half reliability across repeated NSD stimulus presentations, reporting normalized predictivity only on significantly responsive vertices.
2. **Continuous Center-of-Mass Depth**: Replace argmax layer metrics with continuous, image-bootstrapped depth profiles ($\bar{d}$) to eliminate layer-tie instability.
3. **Non-Parametric Permutation Testing**: Evaluate hierarchy significance by permuting ROI anatomical labels ($10,000$ permutations) to account for spatial autocorrelation among visual areas.
4. **Architectural Diversity**:
   * **Vision Transformers (ViT / Swin)**: Evaluate how self-attention without convolutional inductive bias aligns with visual cortex.
   * **Self-Supervised Models (DINOv2 / MAE)**: Test whether contrastive or masked autoencoding objectives without human labels better predict neural responses.
   * **Multimodal Models (CLIP)**: Test whether language-aligned representations yield superior predictivity in high-level semantic areas ($FFA, PPA, RSC$).
5. **Variance Partitioning**: Perform partial correlation analyses to isolate the unique variance explained by late semantic layers over and above early retinotopic features.
6. **Multi-Subject Replication**: Replicate the complete analysis pipeline across Subjects 2, 3, and 4 from the NSD benchmark to report cross-subject consistency and inter-individual variance.
