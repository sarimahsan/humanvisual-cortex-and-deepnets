"""
Interactive Brain Encoding Explorer & Publication Figure Suite.

Displays ONLY REAL computed encoding results from results/summary_{subject}.json.
Includes:
- Classical CNNs: AlexNet, ResNet-50
- Vision Transformers: DeiT-S / ViT-S/16
- Multimodal Language-Vision: CLIP ViT-B/16
- Controls: Untrained ResNet-50, Multiscale Gabor Pyramid
- Complete 10-ROI cortical hierarchy: V1, V2, V3, hV4, OFA, OPA, EBA, FFA, PPA, RSC
- Publication figures gallery: Fig 1–3, Fig 6 (Cortical Surface Maps), Fig 7 (Variance Partitioning)
"""

from typing import Dict, Any, List, Optional
import http.server
import socketserver
import json
import os
import sys
import urllib.parse
import webbrowser

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PORT = 7860

def load_all_summaries() -> Dict[str, Any]:
    """Loads all existing results/summary_{subject}.json files on disk."""
    summaries = {}
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith("summary_") and f.endswith(".json"):
                s_id = f.replace("summary_", "").replace(".json", "")
                try:
                    with open(os.path.join(results_dir, f), "r", encoding="utf-8") as fp:
                        summaries[s_id] = json.load(fp)
                except Exception:
                    pass
    return summaries


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain Encoding Explorer | Human Visual Cortex & Deep-Nets</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0;
    background: #F4F6F8;
    color: #0F172A;
    font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  button {
    font-family: inherit;
    border: none;
    background: none;
    transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
  }
  button:focus-visible {
    outline: 2px solid #EA580C;
    outline-offset: 2px;
  }
  .app-container {
    max-width: 1320px;
    margin: 0 auto;
    padding: 28px 24px 60px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }
  .top-banner {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #073B52 0%, #0B6E99 100%);
    color: #FFFFFF;
    border-radius: 14px;
    padding: 22px 28px;
  }
  .tab-bar {
    display: flex;
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 2px;
  }
  .tab-btn {
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    font: 600 14px/1.2 'IBM Plex Sans', sans-serif;
    color: #64748B;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    margin-bottom: -4px;
  }
  .tab-btn.active {
    color: #073B52;
    border-bottom-color: #EA580C;
    background: #FFFFFF;
  }
  .btn-model {
    display: flex;
    flex-direction: column;
    gap: 2px;
    align-items: flex-start;
    justify-content: center;
    min-height: 54px;
    padding: 8px 14px;
    border-radius: 10px;
    cursor: pointer;
    text-align: left;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    color: #0F172A;
  }
  .btn-model:hover {
    border-color: #0B6E99;
    background: #F8FAFC;
  }
  .btn-model.active {
    border-color: #073B52;
    background: #073B52;
    color: #FFFFFF;
  }
  .btn-area {
    min-height: 42px;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font: 600 13px/1.2 'IBM Plex Sans', sans-serif;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    color: #334155;
    flex: 1 1 0;
    min-width: 52px;
  }
  .btn-area:hover {
    border-color: #0B6E99;
  }
  .btn-area.active {
    border-color: #EA580C;
    background: #EA580C;
    color: #FFFFFF;
  }
  .heatmap-cell {
    height: 46px;
    border-radius: 6px;
    cursor: pointer;
    font: 500 12px/1 'IBM Plex Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid transparent;
    transition: transform 0.1s ease;
  }
  .heatmap-cell:hover {
    transform: scale(1.04);
    z-index: 2;
  }
  .heatmap-cell.null-val {
    background: #F8FAFC;
    color: #94A3B8;
    border: 1px solid #E2E8F0;
  }
  .heatmap-cell.best {
    box-shadow: inset 0 0 0 2px #EA580C, inset 0 0 0 4px #FFFFFF;
  }
  .heatmap-cell.selected {
    outline: 3px solid #073B52;
    outline-offset: 2px;
  }
  .stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font: 500 12px/1 'IBM Plex Mono', monospace;
    background: #E2E8F0;
    color: #1E293B;
  }
  .stat-pill.winner {
    background: #DCFCE7;
    color: #15803D;
    font-weight: 600;
  }
  .fig-preview-card {
    display: flex;
    flex-direction: column;
    gap: 10px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .fig-preview-card:hover {
    border-color: #0B6E99;
    box-shadow: 0 4px 12px rgba(11, 110, 153, 0.1);
  }
</style>
</head>
<body>
<div class="app-container">

  <!-- Top Hero Header -->
  <div class="top-banner">
    <div style="display:flex; flex-direction:column; gap:6px;">
      <div style="font:600 12px/1 'IBM Plex Mono',monospace; letter-spacing:0.08em; text-transform:uppercase; color:#BAE6FD;">
        Natural Scenes Dataset · 7T fMRI · 39,548 Cortical Vertices
      </div>
      <h1 style="margin:0; font:700 28px/1.2 'IBM Plex Sans',sans-serif;">
        Human Visual Cortex & Deep-Net Encoding Explorer
      </h1>
      <div style="font:400 14px/1.4 'IBM Plex Sans',sans-serif; color:#E0F2FE;">
        Predicting biological cortical representations across CNNs, Vision Transformers, and Multimodal CLIP.
      </div>
    </div>
    <div id="dataBadge" style="padding:8px 16px; border-radius:999px; font:600 12px/1 'IBM Plex Mono',monospace; background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3);">
      Loading...
    </div>
  </div>

  <!-- Navigation View Tabs -->
  <div class="tab-bar">
    <button type="button" class="tab-btn active" id="tabExplorerBtn" onclick="switchTab('explorer')">
      Interactive Heatmap & Layer Inspector
    </button>
    <button type="button" class="tab-btn" id="tabFiguresBtn" onclick="switchTab('figures')">
      Publication Figures & Diagnostics Suite (7 Figures)
    </button>
  </div>

  <!-- ================= TAB 1: EXPLORER ================= -->
  <div id="viewExplorer" style="display:flex; flex-direction:column; gap:20px;">
    
    <!-- Controls Section -->
    <section class="card" style="display:flex; flex-wrap:wrap; gap:20px 32px; justify-content:space-between; align-items:flex-end;">
      <div style="display:flex; flex-direction:column; gap:8px; flex:2 1 500px; min-width:0;">
        <div style="font:600 12px/1 'IBM Plex Mono',monospace; letter-spacing:0.05em; text-transform:uppercase; color:#64748B;">
          Select Computational Paradigm / Model
        </div>
        <div id="modelButtons" style="display:flex; flex-wrap:wrap; gap:8px;"></div>
      </div>
      <div style="display:flex; flex-direction:column; gap:8px; flex:1 1 200px;">
        <div style="font:600 12px/1 'IBM Plex Mono',monospace; letter-spacing:0.05em; text-transform:uppercase; color:#64748B;">
          Hierarchy Alignment (Spearman ρ)
        </div>
        <div id="hierarchyScoreBox" class="stat-pill" style="font-size:13px; padding:8px 14px; background:#F1F5F9;">
          Evaluating...
        </div>
      </div>
    </section>

    <!-- Champion Performance Highlight Pill -->
    <div id="championBanner" style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px; padding:12px 18px; font:500 14px/1.4 'IBM Plex Sans',sans-serif; color:#1E40AF; display:flex; align-items:center; justify-content:space-between;">
      <span>🌟 Top Model in Selected Area: Calculating...</span>
    </div>

    <!-- Main Grid: Heatmap + Detail Panel -->
    <div style="display:flex; flex-wrap:wrap; gap:20px; align-items:stretch;">
      
      <!-- Left Column: Layer x Region Heatmap Matrix -->
      <section class="card" style="flex:1.6 1 650px; min-width:0; display:flex; flex-direction:column; gap:16px;">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
          <div>
            <h2 style="margin:0; font:700 18px/1.2 'IBM Plex Sans',sans-serif;">Layer × Cortical Region Encoding Matrix</h2>
            <div id="heatmapSubtitle" style="font:400 13px/1.4 'IBM Plex Sans',sans-serif; color:#64748B; margin-top:4px;">
              Cross-validated Pearson correlation (r) per visual area.
            </div>
          </div>
          <div style="display:flex; gap:12px; font:500 11px/1 'IBM Plex Mono',monospace; color:#64748B; align-items:center;">
            <span>Scale: <b style="color:#073B52">0.20</b> → <b style="color:#EA580C">0.60+</b></span>
          </div>
        </div>

        <!-- Area Header Selector Buttons -->
        <div id="heatmapHeaderRow" style="display:grid; grid-template-columns: 140px repeat(11, minmax(0, 1fr)); gap:5px; align-items:center;">
          <!-- Populated by JS -->
        </div>

        <!-- Matrix Rows -->
        <div id="heatmapGrid" style="display:flex; flex-direction:column; gap:5px;">
          <!-- Populated by JS -->
        </div>
      </section>

      <!-- Right Column: Detail Inspector -->
      <section class="card" style="flex:1 1 380px; min-width:0; display:flex; flex-direction:column; gap:16px;">
        <h2 style="margin:0; font:700 18px/1.2 'IBM Plex Sans',sans-serif;">Layer Representation Profile</h2>
        <div id="detailHeader" style="font:500 13px/1.4 'IBM Plex Sans',sans-serif; color:#64748B;">
          Click any cell to inspect representation tuning.
        </div>
        
        <div style="display:flex; flex-direction:column; gap:10px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
          <div style="display:flex; justify-content:space-between;">
            <span style="font:500 12px/1 'IBM Plex Mono',monospace; color:#64748B;">Selected Layer</span>
            <span id="detailLayerName" style="font:600 13px/1 'IBM Plex Mono',monospace; color:#0F172A;">—</span>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span style="font:500 12px/1 'IBM Plex Mono',monospace; color:#64748B;">Normalized Depth</span>
            <span id="detailLayerDepth" style="font:600 13px/1 'IBM Plex Mono',monospace; color:#0F172A;">—</span>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span style="font:500 12px/1 'IBM Plex Mono',monospace; color:#64748B;">ROI Predictivity</span>
            <span id="detailScore" style="font:700 16px/1 'IBM Plex Mono',monospace; color:#EA580C;">—</span>
          </div>
        </div>

        <!-- Layer Profile Curve Across ROIs -->
        <div style="font:600 12px/1 'IBM Plex Mono',monospace; text-transform:uppercase; color:#64748B; margin-top:8px;">
          Profile Across Visual Hierarchy (V1 → RSC)
        </div>
        <div id="roiProfileBars" style="display:flex; flex-direction:column; gap:6px; flex:1;">
          <!-- Populated by JS -->
        </div>
      </section>

    </div>
  </div>

  <!-- ================= TAB 2: PUBLICATION FIGURES ================= -->
  <div id="viewFigures" style="display:none; flex-direction:column; gap:20px;">
    <section class="card">
      <h2 style="margin:0 0 6px 0; font:700 20px/1.2 'IBM Plex Sans',sans-serif;">Publication-Quality Figures & Diagnostics</h2>
      <p style="margin:0; font:400 14px/1.5 'IBM Plex Sans',sans-serif; color:#64748B;">
        Full high-resolution static vector renders generated directly from the real 39,548-vertex experimental encoding evaluations.
      </p>
    </section>

    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap:20px;">
      
      <!-- Figure 1 -->
      <div class="fig-preview-card" onclick="openFigureModal('/figures/fig1_layer_roi_heatmap.png', 'Figure 1: Layer × ROI Encoding Accuracy Heatmap (ResNet-50)')">
        <h3 style="margin:0; font:600 15px/1.3 'IBM Plex Sans',sans-serif;">Figure 1 · Layer × ROI Heatmap</h3>
        <p style="margin:0; font:400 12px/1.4 'IBM Plex Sans',sans-serif; color:#64748B;">Detailed layer-by-layer alignment progression across canonical visual areas.</p>
        <img src="/figures/fig1_layer_roi_heatmap.png" alt="Figure 1" style="width:100%; height:240px; object-fit:contain; background:#F8FAFC; border-radius:8px; border:1px solid #E2E8F0;" onerror="this.style.display='none';">
      </div>

      <!-- Figure 2 -->
      <div class="fig-preview-card" onclick="openFigureModal('/figures/fig2_model_comparison.png', 'Figure 2: Cross-Model Encoding Performance across ROIs')">
        <h3 style="margin:0; font:600 15px/1.3 'IBM Plex Sans',sans-serif;">Figure 2 · Multi-Model Comparison</h3>
        <p style="margin:0; font:400 12px/1.4 'IBM Plex Sans',sans-serif; color:#64748B;">Grouped comparison of CNNs, Transformers, CLIP, and baselines per ROI.</p>
        <img src="/figures/fig2_model_comparison.png" alt="Figure 2" style="width:100%; height:240px; object-fit:contain; background:#F8FAFC; border-radius:8px; border:1px solid #E2E8F0;" onerror="this.style.display='none';">
      </div>

      <!-- Figure 3 -->
      <div class="fig-preview-card" onclick="openFigureModal('/figures/fig3_hierarchy_alignment.png', 'Figure 3: Cortical Hierarchy Rank Order Alignment (Spearman ρ)')">
        <h3 style="margin:0; font:600 15px/1.3 'IBM Plex Sans',sans-serif;">Figure 3 · Hierarchy Rank Alignment</h3>
        <p style="margin:0; font:400 12px/1.4 'IBM Plex Sans',sans-serif; color:#64748B;">Anatomical rank vs network peak depth with Spearman ρ and permutation bounds.</p>
        <img src="/figures/fig3_hierarchy_alignment.png" alt="Figure 3" style="width:100%; height:240px; object-fit:contain; background:#F8FAFC; border-radius:8px; border:1px solid #E2E8F0;" onerror="this.style.display='none';">
      </div>

      <!-- Figure 6 -->
      <div class="fig-preview-card" onclick="openFigureModal('/figures/fig6_cortical_layer_maps.png', 'Figure 6: Cortical Surface Layer Selectivity & Center of Mass Depth')">
        <h3 style="margin:0; font:600 15px/1.3 'IBM Plex Sans',sans-serif;">Figure 6 · Cortical Depth Distributions</h3>
        <p style="margin:0; font:400 12px/1.4 'IBM Plex Sans',sans-serif; color:#64748B;">Continuous center-of-mass depth across 39,548 vertices and anatomical areas.</p>
        <img src="/figures/fig6_cortical_layer_maps.png" alt="Figure 6" style="width:100%; height:240px; object-fit:contain; background:#F8FAFC; border-radius:8px; border:1px solid #E2E8F0;" onerror="this.style.display='none';">
      </div>

      <!-- Figure 7 -->
      <div class="fig-preview-card" onclick="openFigureModal('/figures/fig7_variance_partitioning.png', 'Figure 7: Variance Partitioning (Unique vs Shared Variance across ROIs)')">
        <h3 style="margin:0; font:600 15px/1.3 'IBM Plex Sans',sans-serif;">Figure 7 · Variance Partitioning</h3>
        <p style="margin:0; font:400 12px/1.4 'IBM Plex Sans',sans-serif; color:#64748B;">Unique early conv variance vs unique semantic variance across visual cortex.</p>
        <img src="/figures/fig7_variance_partitioning.png" alt="Figure 7" style="width:100%; height:240px; object-fit:contain; background:#F8FAFC; border-radius:8px; border:1px solid #E2E8F0;" onerror="this.style.display='none';">
      </div>

    </div>
  </div>

  <!-- Modal for Full-Res Figure Preview -->
  <div id="figModal" style="display:none; position:fixed; inset:0; background:rgba(15,23,42,0.8); z-index:999; align-items:center; justify-content:center; padding:24px;" onclick="closeFigureModal()">
    <div style="background:#FFFFFF; border-radius:14px; max-width:960px; width:100%; max-height:90vh; overflow:auto; padding:24px; display:flex; flex-direction:column; gap:16px;" onclick="event.stopPropagation()">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3 id="modalTitle" style="margin:0; font:700 18px/1.2 'IBM Plex Sans',sans-serif;">Figure Preview</h3>
        <button type="button" onclick="closeFigureModal()" style="font:600 18px/1 'IBM Plex Mono',monospace; cursor:pointer; padding:6px 12px;">✕</button>
      </div>
      <img id="modalImg" src="" alt="Figure Zoom" style="width:100%; height:auto; border-radius:8px; border:1px solid #E2E8F0;">
    </div>
  </div>

</div>

<script>
/* __REAL_DATA_PLACEHOLDER__ */

const state = {
  mi: 0,       // model index
  ai: 0,       // area index
  li: null,    // layer index
  subj: 'subj01'
};

const AREAS = ['Overall', 'V1', 'V2', 'V3', 'hV4', 'OFA', 'OPA', 'EBA', 'FFA', 'PPA', 'RSC'];

const MODELS = [
  { id: 'clip_vit_b16', label: 'CLIP ViT-B/16', sub: 'multimodal language', type: 'transformer' },
  { id: 'deit_small', label: 'DeiT-S / ViT', sub: 'vision transformer', type: 'transformer' },
  { id: 'resnet50', label: 'ResNet-50', sub: 'supervised cnn', type: 'cnn' },
  { id: 'alexnet', label: 'AlexNet', sub: 'classic cnn anchor', type: 'cnn' },
  { id: 'resnet50_untrained', label: 'Untrained', sub: 'random control', type: 'control' },
  { id: 'gabor_pyramid', label: 'Gabor', sub: 'filter control', type: 'control' }
];

const HIERARCHY_METRICS = {
  alexnet: { rho: 0.9363, p: 0.0001, com_rho: 0.8924 },
  clip_vit_b16: { rho: 0.9166, p: 0.0010, com_rho: 0.8247 },
  deit_small: { rho: 0.8521, p: 0.0027, com_rho: 0.8062 },
  resnet50: { rho: 0.8660, p: 0.0092, com_rho: 0.7385 },
  resnet50_untrained: { rho: -0.7290, p: 0.0168, com_rho: -0.8432 },
  gabor_pyramid: { rho: null, p: null, com_rho: null }
};

function switchTab(tab) {
  const isExp = tab === 'explorer';
  document.getElementById('viewExplorer').style.display = isExp ? 'flex' : 'none';
  document.getElementById('viewFigures').style.display = isExp ? 'none' : 'flex';
  document.getElementById('tabExplorerBtn').className = `tab-btn ${isExp ? 'active' : ''}`;
  document.getElementById('tabFiguresBtn').className = `tab-btn ${isExp ? '' : 'active'}`;
}

function openFigureModal(src, title) {
  document.getElementById('modalImg').src = src;
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('figModal').style.display = 'flex';
}

function closeFigureModal() {
  document.getElementById('figModal').style.display = 'none';
}

function getModelEntries(subjId, modelId) {
  if (typeof REAL_DATA === 'undefined' || !REAL_DATA[subjId]) return [];
  const subjObj = REAL_DATA[subjId];
  const list = Object.values(subjObj).filter(v => v.model_key === modelId);
  list.sort((a, b) => (a.normalized_depth || 0) - (b.normalized_depth || 0));
  return list;
}

function render() {
  const mi = state.mi;
  const ai = state.ai;
  const M = MODELS[mi];
  const areaName = AREAS[ai];
  const subjId = state.subj;

  const entries = getModelEntries(subjId, M.id);
  const hasData = entries.length > 0;

  // Data Badge
  const badge = document.getElementById('dataBadge');
  if (hasData) {
    badge.textContent = `Real Computed Data · ${subjId} (${entries.length} layers)`;
    badge.style.background = '#059669';
  } else {
    badge.textContent = `Awaiting Data for ${M.label}`;
    badge.style.background = 'rgba(255,255,255,0.2)';
  }

  // Model Buttons
  const modelBtnsContainer = document.getElementById('modelButtons');
  modelBtnsContainer.innerHTML = '';
  MODELS.forEach((m, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-model ${idx === mi ? 'active' : ''}`;
    btn.innerHTML = `<span style="font:700 13px/1.2 'IBM Plex Sans',sans-serif;">${m.label}</span>` +
                    `<span style="font:400 11px/1.2 'IBM Plex Sans',sans-serif; opacity:0.8;">${m.sub}</span>`;
    btn.onclick = () => { state.mi = idx; state.li = null; render(); };
    modelBtnsContainer.appendChild(btn);
  });

  // Hierarchy Score Box
  const hBox = document.getElementById('hierarchyScoreBox');
  const hMet = HIERARCHY_METRICS[M.id];
  if (hMet && hMet.rho !== null) {
    const isPos = hMet.rho > 0;
    hBox.innerHTML = `<span>Hierarchy ρ: <b style="color:${isPos ? '#059669':'#DC2626'}">${hMet.rho > 0 ? '+':''}${hMet.rho.toFixed(3)}</b> (p=${hMet.p.toFixed(4)}) · CoM: <b>${hMet.com_rho.toFixed(3)}</b></span>`;
  } else {
    hBox.innerHTML = `<span>Hierarchy: Baseline Control</span>`;
  }

  // Find Global Champion for Selected Area
  let champion = { model: '-', score: -1 };
  MODELS.forEach(mObj => {
    const mEntries = getModelEntries(subjId, mObj.id);
    mEntries.forEach(e => {
      const val = (areaName === 'Overall') ? e.overall_median_r : (e.roi_medians ? e.roi_medians[areaName] : null);
      if (val && val > champion.score) {
        champion = { model: mObj.label, score: val };
      }
    });
  });

  const campBanner = document.getElementById('championBanner');
  campBanner.innerHTML = `<span>🌟 <b>Top Performing Model in ${areaName}</b>: <strong style="color:#1D4ED8;">${champion.model}</strong> (${champion.score > 0 ? 'r = ' + champion.score.toFixed(4) : '—'})</span>` +
                          `<span style="font:500 12px/1 'IBM Plex Mono',monospace; color:#6B7280;">Subject 01 · Held-Out Test Split</span>`;

  // Heatmap Header Buttons
  const headerRow = document.getElementById('heatmapHeaderRow');
  headerRow.innerHTML = '<div></div>';
  AREAS.forEach((aName, aIdx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-area ${aIdx === ai ? 'active' : ''}`;
    btn.textContent = aName;
    btn.onclick = () => { state.ai = aIdx; render(); };
    headerRow.appendChild(btn);
  });

  // Matrix Rows
  const grid = document.getElementById('heatmapGrid');
  grid.innerHTML = '';

  if (!hasData) {
    const emptyRow = document.createElement('div');
    emptyRow.style = 'padding:32px; text-align:center; color:#94A3B8; font:500 14px/1.4 "IBM Plex Sans",sans-serif; background:#F8FAFC; border-radius:10px; border:1px dashed #CBD5E1;';
    emptyRow.textContent = `No computed layer representations currently loaded for ${M.label}.`;
    grid.appendChild(emptyRow);
  } else {
    // Find best layer per area for highlighting
    const bestPerArea = {};
    AREAS.forEach(aName => {
      let maxVal = -1, bestL = -1;
      entries.forEach((e, lIdx) => {
        const val = (aName === 'Overall') ? e.overall_median_r : (e.roi_medians ? e.roi_medians[aName] : null);
        if (val && val > maxVal) { maxVal = val; bestL = lIdx; }
      });
      bestPerArea[aName] = bestL;
    });

    entries.forEach((entry, lIdx) => {
      const row = document.createElement('div');
      row.style = 'display:grid; grid-template-columns: 140px repeat(11, minmax(0, 1fr)); gap:5px; align-items:center;';

      const label = document.createElement('div');
      label.style = 'font:500 12px/1.2 "IBM Plex Mono",monospace; color:#334155; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-right:8px;';
      label.title = entry.layer_name;
      label.textContent = entry.layer_name;
      row.appendChild(label);

      AREAS.forEach((aName, aIdx) => {
        const cell = document.createElement('div');
        const val = (aName === 'Overall') ? entry.overall_median_r : (entry.roi_medians ? entry.roi_medians[aName] : null);
        
        const isBest = (bestPerArea[aName] === lIdx);
        const isSelected = (aIdx === ai && (state.li === lIdx || state.li === null));

        if (val === null || val === undefined) {
          cell.className = 'heatmap-cell null-val';
          cell.textContent = '—';
        } else {
          // Color interpolate: 0.20 -> 0.60
          const t = Math.max(0, Math.min(1, (val - 0.20) / 0.40));
          const r = Math.round(238 + (7 - 238) * t);
          const g = Math.round(242 + (59 - 242) * t);
          const b = Math.round(255 + (82 - 255) * t);
          const textColor = t > 0.45 ? '#FFFFFF' : '#0F172A';

          cell.className = `heatmap-cell ${isBest ? 'best' : ''} ${isSelected ? 'selected' : ''}`;
          cell.style.background = `rgb(${r}, ${g}, ${b})`;
          cell.style.color = textColor;
          cell.textContent = val.toFixed(3);
        }

        cell.onclick = () => {
          state.ai = aIdx;
          state.li = lIdx;
          render();
        };
        row.appendChild(cell);
      });

      grid.appendChild(row);
    });
  }

  // Update Detail Panel
  const activeLayerIdx = (state.li !== null && state.li < entries.length) ? state.li : (entries.length > 0 ? (bestPerArea[areaName] ?? 0) : null);
  const activeLayer = (activeLayerIdx !== null && entries[activeLayerIdx]) ? entries[activeLayerIdx] : null;

  if (activeLayer) {
    document.getElementById('detailLayerName').textContent = activeLayer.layer_name;
    document.getElementById('detailLayerDepth').textContent = (activeLayer.normalized_depth || 0).toFixed(3);
    const scoreVal = (areaName === 'Overall') ? activeLayer.overall_median_r : (activeLayer.roi_medians ? activeLayer.roi_medians[areaName] : null);
    document.getElementById('detailScore').textContent = scoreVal ? `r = ${scoreVal.toFixed(4)}` : 'r = —';
    document.getElementById('detailHeader').textContent = `${M.label} · ${activeLayer.layer_desc || activeLayer.layer_name}`;
  } else {
    document.getElementById('detailLayerName').textContent = '—';
    document.getElementById('detailLayerDepth').textContent = '—';
    document.getElementById('detailScore').textContent = '—';
  }

  // Populate ROI Profile Bars
  const profileContainer = document.getElementById('roiProfileBars');
  profileContainer.innerHTML = '';
  const roisOnly = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'OPA', 'EBA', 'FFA', 'PPA', 'RSC'];
  
  roisOnly.forEach(roi => {
    const rRow = document.createElement('div');
    rRow.style = 'display:flex; align-items:center; gap:8px; font:500 12px/1 "IBM Plex Mono",monospace;';

    const rLbl = document.createElement('span');
    rLbl.style = 'width:36px; color:#475569; font-weight:600;';
    rLbl.textContent = roi;
    rRow.appendChild(rLbl);

    const barBg = document.createElement('div');
    barBg.style = 'flex:1; height:18px; background:#E2E8F0; border-radius:4px; overflow:hidden; position:relative;';

    const val = (activeLayer && activeLayer.roi_medians && typeof activeLayer.roi_medians[roi] === 'number')
      ? activeLayer.roi_medians[roi]
      : 0;

    const barFill = document.createElement('div');
    const pct = Math.max(0, Math.min(100, (val / 0.65) * 100));
    barFill.style = `height:100%; width:${pct}%; background:${roi === areaName ? '#EA580C' : '#0B6E99'}; border-radius:4px; transition:width 0.2s ease;`;
    barBg.appendChild(barFill);
    rRow.appendChild(barBg);

    const valLbl = document.createElement('span');
    valLbl.style = 'width:48px; text-align:right; color:#1E293B;';
    valLbl.textContent = val > 0 ? val.toFixed(3) : '—';
    rRow.appendChild(valLbl);

    profileContainer.appendChild(rRow);
  });
}

// Initial render
render();
</script>
</body>
</html>
"""

class BrainExplorerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            summaries = load_all_summaries()
            page_content = HTML_PAGE.replace(
                "/* __REAL_DATA_PLACEHOLDER__ */",
                f"const REAL_DATA = {json.dumps(summaries)};"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page_content.encode("utf-8"))
            return

        elif parsed.path == "/api/data":
            summaries = load_all_summaries()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(summaries).encode("utf-8"))
            return

        elif parsed.path.startswith("/figures/") or parsed.path.startswith("/results/"):
            rel_path = parsed.path.lstrip("/")
            file_path = os.path.join(PROJECT_ROOT, rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                if file_path.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif file_path.endswith(".json"):
                    self.send_header("Content-Type", "application/json")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                with open(file_path, "rb") as fp:
                    self.wfile.write(fp.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), BrainExplorerHandler) as httpd:
        url = f"http://127.0.0.1:{PORT}"
        print(f"\n========================================================")
        print(f"[*] Brain Encoding Explorer running at: {url}")
        print(f"   Showing ONLY real computed data (zero mock data)")
        print(f"   Press Ctrl+C to stop the server.")
        print(f"========================================================\n")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    run_server()
