# Brain Encoding Project: Which Deep-Net Layers Explain Which Human Visual Regions?

A reproduction-and-extension computational neuroimaging pipeline evaluating how deep neural network layer representations predict human visual cortex responses (fMRI) across the visual hierarchy.

Designed to run cleanly and reproducibly on single T4 GPUs (Colab / Kaggle) using the **Algonauts 2023 (NSD)** public dataset release.

---

## 🔬 Core Questions

1. **Hierarchy**: Do early network layers best predict early visual cortex (V1–V3), while deeper layers best predict higher ventral visual regions (hV4, category-selective areas)?
2. **Training Objective**: How do supervised (DeiT-S), self-supervised (DINO ViT-S/16), and language-image models (CLIP ViT-B/16) align across cortical regions when architecture is controlled?
3. **Learned vs. Built-in**: How much alignment stems from task-driven learning vs. architecture inductive biases (untrained ResNet-50) or low-level pixel/spatial-frequency statistics (Gabor pyramid)?
4. **Category Selectivity**: Do predicted responses in category-selective ROIs mirror human domain preferences (faces, bodies, places, food, words) for held-out natural images?
5. **Robustness**: Are hierarchy and model rankings preserved across individual human subjects (Subjects 1–4+)?

---

## 📂 Repository Structure

```
├── configs/               # Model taps, layer definitions, hyperparameter grids
│   ├── experiment.yaml    # Global parameters, PCA dims, alpha grid, splits
│   └── models.yaml        # Taps, pooling rules, normalization depth specs
├── data/                  # Data loaders, split generation, ROI mapping, COCO labels
│   ├── algonauts.py       # Algonauts 2023 format loader & split manager
│   ├── rois.py            # Cortical surface ROIs & groupings (Early, Mid, High)
│   ├── coco_labels.py     # COCO 80-category mapping & supercategory parser
│   └── mock_data.py       # Lightweight synthetic generator for local test runs
├── features/              # Feature extraction, spatial pooling, and caching
│   ├── extract.py         # Multi-model extractor with standardized taps
│   ├── gabor.py           # Low-level baseline: Gabor wavelet pyramid + color
│   └── cache.py           # Float16 / memory-mapped array caching
├── encoding/              # Encoding models & cross-validation
│   ├── ridge.py           # SVD / kernelized Ridge regression (GPU & CPU)
│   ├── pca.py             # Train-only PCA dimensionality reduction
│   └── pipeline.py        # End-to-end encoding model runner per subject
├── analysis/              # Downstream neuroscientific analyses
│   ├── hierarchy.py       # Peak layer depth vs ROI order (Spearman ρ)
│   ├── selectivity.py     # Predicted vs measured category selectivity profiles
│   ├── comparison.py      # Paired bootstrap tests between models
│   └── plotting.py        # Publication-ready static figures
├── notebooks/             # Ready-to-run Jupyter / Colab notebooks
│   └── colab_runner.ipynb # Self-contained Google Colab / Kaggle pipeline
├── writeup/               # Pre-registration and scientific report
│   ├── preregistration.md # Pre-registered hypotheses (committed before results)
│   └── report.md          # 3-5 page writeup & honest scientific conclusions
├── app/                   # Interactive Gradio explorer
│   └── app.py             # Interactive dashboard (ROIs, layers, selectivity)
├── tests/                 # Unit and integration test suite
│   └── test_pipeline.py   # End-to-end sanity check with synthetic data
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/your-username/human-visual-cortex-and-deepnets.git
cd human-visual-cortex-and-deepnets
pip install -r requirements.txt
```

### 2. Verify Pipeline on Synthetic Data (No Download Needed)
You can run an end-to-end sanity check immediately:
```bash
python -m tests.test_pipeline
```

### 3. Running with Algonauts 2023 NSD Data
Download subject data following the [Algonauts 2023 Challenge guide](https://algonautsproject.csail.mit.edu/):
```
data/raw/algonauts_2023/
  ├── subj01/
  │   ├── training_split/
  │   │   ├── training_images/
  │   │   └── training_fmri/
  │   │       ├── lh_training_fmri.npy
  │   │       └── rh_training_fmri.npy
  │   └── roi_masks/
  ...
```

Run feature extraction:
```bash
python -m features.extract --config configs/models.yaml --subject subj01
```

Fit ridge encoding models & evaluate:
```bash
python -m encoding.pipeline --subject subj01
```

Generate figures:
```bash
python -m analysis.plotting --output_dir figures/
```

Launch the interactive explorer:
```bash
python app/app.py
```

---

## 📊 Models & Controls

| Model | Architecture | Objective | Key Purpose |
|---|---|---|---|
| **AlexNet** | CNN (8 layers) | Supervised (ImageNet-1k) | Algonauts baseline & benchmark anchor |
| **ResNet-50** | CNN (50 layers) | Supervised (ImageNet-1k) | Standard modern CNN benchmark |
| **DeiT-S / ViT-S/16** | Vision Transformer | Supervised (ImageNet-1k) | Direct architectural match to DINO |
| **DINO ViT-S/16** | Vision Transformer | Self-Supervised | Isolates training objective vs. DeiT-S |
| **CLIP ViT-B/16** | Vision Transformer | Language-Image Contrastive | Tests multimodality / semantic alignment |
| **Untrained ResNet-50** | CNN (random weights) | None (Random init) | Control: Architecture & inductive bias |
| **Gabor Pyramid** | Wavelet Filter Bank | Classical Vision | Control: Low-level spatial & color statistics |

---

## 📜 Scientific Rigor & Pre-registration
Prior to inspecting final results, all core hypotheses were pre-registered in [`writeup/preregistration.md`](file:///d:/human-visual-cortex-and-deepnets/writeup/preregistration.md).
See [`writeup/report.md`](file:///d:/human-visual-cortex-and-deepnets/writeup/report.md) for full methods, cross-subject findings, and critical discussions on interpretability limits.
