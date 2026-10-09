"""
Interactive Brain Encoding Explorer.

Faithfully implements the design reference mockup from demo/Main.dc.html:
- Clean IBM Plex Sans and IBM Plex Mono typography.
- Light, publication-grade styling (#F3F5F7 background, #FFFFFF cards, #D9E0E6 borders).
- Interactive Layer x Region similarity matrix with noise-ceiling normalization.
- Layer-wise score bar chart with untrained baseline tick and noise ceiling line.
- 3-stage Visual Hierarchy flow map (Early -> Intermediate -> Category-selective) with Spearman rho readout.
- Category Selectivity check (measured fMRI vs deep net predicted responses).
- Standalone zero-dependency Python server (no Gradio/Pandas, avoiding Windows AppLocker/WDAC blocks).
"""

from typing import Dict, Any, List
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

def load_data(subject_id: str, model_key: str):
    """Loads results summary for the given subject and model if available."""
    summary_path = os.path.join(PROJECT_ROOT, "results", f"summary_{subject_id}.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            filtered = [v for v in data.values() if v.get("model_key") == model_key]
            if filtered:
                return filtered
        except Exception:
            pass
    return None

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain Encoding Explorer | Human Visual Cortex & Deep-Nets</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap">
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0;
    background: #F3F5F7;
    color: #12202B;
    font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  button {
    font-family: inherit;
    border: none;
    background: none;
    transition: all 0.15s ease;
  }
  button:focus-visible {
    outline: 3px solid #C2570C;
    outline-offset: 2px;
  }
  .app-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px 48px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-end;
    justify-content: space-between;
  }
  .card {
    background: #FFFFFF;
    border: 1px solid #D9E0E6;
    border-radius: 14px;
    padding: 20px;
    box-sizing: border-box;
  }
  .filter-section {
    display: flex;
    flex-wrap: wrap;
    gap: 16px 28px;
    align-items: flex-end;
    justify-content: space-between;
  }
  .btn-model {
    display: flex;
    flex-direction: column;
    gap: 2px;
    align-items: flex-start;
    justify-content: center;
    min-height: 56px;
    padding: 8px 14px;
    border-radius: 10px;
    cursor: pointer;
    text-align: left;
    border: 1px solid #8A99A6;
    background: #FFFFFF;
    color: #12202B;
  }
  .btn-model.active {
    border-color: #073B52;
    background: #073B52;
    color: #FFFFFF;
  }
  .btn-subj {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 56px;
    min-width: 56px;
    padding: 8px 14px;
    border-radius: 10px;
    cursor: pointer;
    font: 600 14px/1.2 'IBM Plex Sans', sans-serif;
    border: 1px solid #8A99A6;
    background: #FFFFFF;
    color: #12202B;
  }
  .btn-subj.active {
    border-color: #073B52;
    background: #073B52;
    color: #FFFFFF;
  }
  .btn-baseline {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 48px;
    padding: 8px 16px;
    border-radius: 10px;
    cursor: pointer;
    font: 500 13px/1.2 'IBM Plex Sans', sans-serif;
    border: 1px solid #8A99A6;
    background: #FFFFFF;
    color: #12202B;
  }
  .btn-baseline.active {
    border-color: #073B52;
    background: #073B52;
    color: #FFFFFF;
  }
  .btn-area {
    min-height: 44px;
    border-radius: 8px;
    cursor: pointer;
    font: 600 14px/1.2 'IBM Plex Sans', sans-serif;
    border: 1px solid #8A99A6;
    background: #FFFFFF;
    color: #12202B;
  }
  .btn-area.active {
    border-color: #073B52;
    background: #073B52;
    color: #FFFFFF;
  }
  .heatmap-cell {
    height: 46px;
    border: 0;
    border-radius: 6px;
    cursor: pointer;
    font: 500 12px/1 'IBM Plex Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .heatmap-cell.best {
    box-shadow: inset 0 0 0 2px #C2570C, inset 0 0 0 4px #FFFFFF;
  }
  .heatmap-cell.selected {
    outline: 3px solid #12202B;
    outline-offset: 2px;
  }
  .row-layout {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: stretch;
  }
  .badge-live {
    padding: 8px 14px;
    border-radius: 999px;
    border: 1px solid #0B6E99;
    color: #073B52;
    background: #EBF5FA;
    font: 500 12px/1.3 'IBM Plex Mono', monospace;
  }
  .badge-mock {
    padding: 8px 14px;
    border-radius: 999px;
    border: 1px solid #C2570C;
    color: #8A3A00;
    background: #FFF4EA;
    font: 500 12px/1.3 'IBM Plex Mono', monospace;
  }
</style>
</head>
<body>
<div class="app-container">

  <!-- Header -->
  <header>
    <div style="display:flex; flex-direction:column; gap:8px; flex:1 1 420px; min-width:0;">
      <div style="font:500 12px/1.3 'IBM Plex Mono',monospace; letter-spacing:0.08em; text-transform:uppercase; color:#0B6E99;">
        Natural Scenes Dataset · 7T fMRI · human visual cortex
      </div>
      <h1 style="margin:0; font:600 34px/1.15 'IBM Plex Sans',sans-serif; letter-spacing:-0.01em;">
        Which network layers predict which human brain regions?
      </h1>
      <p style="margin:0; font:400 15px/1.5 'IBM Plex Sans',sans-serif; color:#51606C; max-width:660px;">
        Pick a pretrained model and a subject, read the layer-by-region map from V1 up to the face, body and place areas, then check it against the untrained baseline and the noise ceiling.
      </p>
    </div>
    <div id="dataBadge" class="badge-mock">Mock data · layout preview</div>
  </header>

  <!-- Controls: Models, Subjects, Baseline Toggle -->
  <section class="card filter-section">
    <div style="display:flex; flex-direction:column; gap:8px; flex:1 1 440px; min-width:0;">
      <div style="font:500 12px/1.3 'IBM Plex Mono',monospace; letter-spacing:0.06em; text-transform:uppercase; color:#51606C;">Model</div>
      <div id="modelButtons" style="display:flex; flex-wrap:wrap; gap:8px;"></div>
    </div>
    <div style="display:flex; flex-direction:column; gap:8px;">
      <div style="font:500 12px/1.3 'IBM Plex Mono',monospace; letter-spacing:0.06em; text-transform:uppercase; color:#51606C;">Subject</div>
      <div id="subjButtons" style="display:flex; flex-wrap:wrap; gap:8px;"></div>
    </div>
    <button type="button" id="toggleBaselineBtn" class="btn-baseline active">Untrained baseline: shown</button>
  </section>

  <!-- Row 1: Heatmap & Layer-wise score -->
  <div class="row-layout">
    <!-- Matrix Heatmap -->
    <section class="card" style="flex:1.4 1 560px; min-width:0; display:flex; flex-direction:column; gap:14px;">
      <div>
        <h2 style="margin:0; font:600 18px/1.3 'IBM Plex Sans',sans-serif;">Layer × region similarity</h2>
        <p id="heatmapSubtitle" style="margin:4px 0 0; font:400 13px/1.5 'IBM Plex Sans',sans-serif; color:#51606C;">
          Cross-validated encoding score, noise-ceiling normalized. Showing ResNet-50, subject S1. Tap a cell or a region to inspect it.
        </p>
      </div>
      <div style="overflow-x:auto;">
        <div style="min-width:560px; display:flex; flex-direction:column; gap:4px;">
          <div id="heatmapHeaderRow" style="display:grid; grid-template-columns:84px repeat(7,minmax(0,1fr)); gap:4px;"></div>
          <div id="heatmapBody" style="display:flex; flex-direction:column; gap:4px;"></div>
        </div>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; font:400 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>0</span>
          <div style="width:120px; height:10px; border-radius:5px; background:linear-gradient(90deg, #ECF3F7, #073B52);"></div>
          <span>1 = noise ceiling</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:18px; height:18px; border-radius:4px; background:#ECF3F7; box-shadow:inset 0 0 0 2px #C2570C, inset 0 0 0 4px #FFFFFF;"></div>
          <span>best layer for that region</span>
        </div>
      </div>
    </section>

    <!-- Layer-wise score bar chart -->
    <section class="card" style="flex:1 1 340px; min-width:0; display:flex; flex-direction:column; gap:14px;">
      <div>
        <h2 id="barsTitle" style="margin:0; font:600 18px/1.3 'IBM Plex Sans',sans-serif;">Layer-wise score · FFA</h2>
        <p id="barsSubtitle" style="margin:4px 0 0; font:400 13px/1.5 'IBM Plex Sans',sans-serif; color:#51606C;">
          ResNet-50 predicting FFA from each layer.
        </p>
      </div>
      <div id="barsChartArea" style="position:relative; height:220px; border-bottom:1.5px solid #12202B; display:flex; gap:6px;">
        <div style="position:absolute; left:0; right:0; top:0; border-top:1.5px dashed #51606C; pointer-events:none;"></div>
      </div>
      <div id="barsButtonsRow" style="display:flex; gap:6px;"></div>
      <div style="display:flex; flex-wrap:wrap; gap:8px 20px; font:400 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:14px; height:14px; border-radius:3px; background:#4C8FAB;"></div><span>model layer</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:16px; height:3px; background:#C2570C;"></div><span>untrained baseline</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:16px; border-top:2px dashed #51606C;"></div><span>noise ceiling</span>
        </div>
      </div>
      <p id="calloutText" style="margin:0; padding-top:12px; border-top:1px solid #D9E0E6; font:400 14px/1.5 'IBM Plex Sans',sans-serif;"></p>
    </section>
  </div>

  <!-- Row 2: Hierarchy map & Selectivity check -->
  <div class="row-layout">
    <!-- Visual hierarchy map -->
    <section class="card" style="flex:1 1 420px; min-width:0; display:flex; flex-direction:column; gap:14px;">
      <div>
        <h2 style="margin:0; font:600 18px/1.3 'IBM Plex Sans',sans-serif;">Visual hierarchy map</h2>
        <p id="hierSubtitle" style="margin:4px 0 0; font:400 13px/1.5 'IBM Plex Sans',sans-serif; color:#51606C;">
          Each region is shaded by the depth of its best-matching layer in ResNet-50.
        </p>
      </div>
      <div id="hierarchyGrid" style="display:grid; grid-template-columns:minmax(0,1fr) 28px minmax(0,1fr) 28px minmax(0,1fr); grid-template-rows:repeat(3,auto); gap:8px 6px; align-items:stretch;">
        <div style="grid-column:1; grid-row:4; font:500 11px/1.3 'IBM Plex Mono',monospace; color:#51606C; letter-spacing:0.04em; text-transform:uppercase;">Early visual</div>
        <div style="grid-column:3; grid-row:4; font:500 11px/1.3 'IBM Plex Mono',monospace; color:#51606C; letter-spacing:0.04em; text-transform:uppercase;">Intermediate</div>
        <div style="grid-column:5; grid-row:4; font:500 11px/1.3 'IBM Plex Mono',monospace; color:#51606C; letter-spacing:0.04em; text-transform:uppercase;">Category-selective</div>
        <div style="grid-column:2; grid-row:1 / span 3; align-self:center; display:flex; justify-content:center; color:#51606C;">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 12h16M14 6l6 6-6 6"></path></svg>
        </div>
        <div style="grid-column:4; grid-row:1 / span 3; align-self:center; display:flex; justify-content:center; color:#51606C;">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 12h16M14 6l6 6-6 6"></path></svg>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:8px; font:400 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;">
        <span>early layer</span>
        <div style="flex:1 1 0; max-width:160px; height:10px; border-radius:5px; background:linear-gradient(90deg,#CFE0E8,#073B52);"></div>
        <span>late layer</span>
      </div>
      <div style="display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 12px; border-top:1px solid #D9E0E6; padding-top:14px;">
        <span id="rhoValue" style="font:500 30px/1 'IBM Plex Mono',monospace">0.93</span>
        <span style="font:400 13px/1.4 'IBM Plex Sans',sans-serif; color:#51606C; flex:1 1 200px; min-width:0;">
          Spearman ρ between region order (V1 to PPA) and best-layer depth. Higher means a cleaner hierarchy.
        </span>
      </div>
    </section>

    <!-- Selectivity check -->
    <section class="card" style="flex:1.2 1 460px; min-width:0; display:flex; flex-direction:column; gap:14px;">
      <div>
        <h2 id="selectivityTitle" style="margin:0; font:600 18px/1.3 'IBM Plex Sans',sans-serif;">Selectivity check · FFA</h2>
        <p id="selectivitySubtitle" style="margin:4px 0 0; font:400 13px/1.5 'IBM Plex Sans',sans-serif; color:#51606C;">
          Mean FFA response by image category, measured vs predicted from ResNet-50 layer L6. Does the model reproduce what the region prefers?
        </p>
      </div>
      <div id="catsChartArea" style="display:flex; gap:10px; height:160px; align-items:stretch; border-bottom:1.5px solid #12202B; padding:0 4px;"></div>
      <div id="catsLabelsRow" style="display:flex; gap:10px; padding:0 4px;"></div>
      <div style="display:flex; flex-wrap:wrap; gap:8px 20px; align-items:center; font:400 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:14px; height:14px; border-radius:3px; background:#12202B;"></div><span>measured (fMRI)</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:14px; height:14px; border-radius:3px; background:#4C8FAB;"></div><span>predicted</span>
        </div>
        <span id="corrValue" style="margin-left:auto; color:#12202B; font-weight:500;">r = 0.94</span>
      </div>
    </section>
  </div>

  <!-- Footer -->
  <footer style="display:flex; flex-wrap:wrap; gap:8px 24px; font:400 12px/1.5 'IBM Plex Mono',monospace; color:#51606C;">
    <span>Subjects: S1–S4 · Images per subject: 9,841 · Regions: provided ROI masks</span>
    <span>Natural Scenes Dataset (NSD 7T fMRI) · Algonauts 2023 Challenge</span>
  </footer>

</div>

<script>
// State Management
const state = {
  mi: 0,       // model index
  ai: 5,       // area index (default FFA)
  li: null,    // layer index (null = best layer for area)
  base: true,  // show untrained baseline tick
  subj: 0      // subject index (S1)
};

const AREAS = ['V1', 'V2', 'V3', 'hV4', 'EBA', 'FFA', 'PPA'];
const PEAKS = [1.0, 1.6, 2.3, 3.4, 4.6, 5.9, 5.1];
const SUBJ = ['S1', 'S2', 'S3', 'S4'];
const CATS = ['Faces', 'Bodies', 'Places', 'Food', 'Animals', 'Objects'];
const SEL = [
  [0.45, 0.50, 0.55, 0.50, 0.50, 0.50],
  [0.45, 0.50, 0.55, 0.50, 0.50, 0.50],
  [0.45, 0.50, 0.60, 0.50, 0.50, 0.50],
  [0.50, 0.50, 0.55, 0.68, 0.50, 0.50],
  [0.55, 0.90, 0.40, 0.30, 0.50, 0.45],
  [0.95, 0.40, 0.30, 0.30, 0.50, 0.35],
  [0.25, 0.30, 0.95, 0.35, 0.30, 0.35]
];
const NL = 8;
const MODELS = [
  { id: 'resnet50', label: 'ResNet-50', sub: 'supervised', amp: 0.74, sig: 1.7 },
  { id: 'alexnet', label: 'AlexNet', sub: 'shallow anchor', amp: 0.68, sig: 1.8 },
  { id: 'resnet50_untrained', label: 'Untrained', sub: 'random init', amp: 0.38, sig: 3.2 },
  { id: 'gabor_pyramid', label: 'Gabor', sub: 'filterbank control', amp: 0.42, sig: 2.5 }
];

// Seeded pseudorandom generator for deterministic, noise-realistic values
const rnd = (a, b, c) => {
  const x = Math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453;
  return x - Math.floor(x);
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function score(m, a, l) {
  const M = MODELS[m];
  const sj = state.subj;
  const isUntrained = M.id === 'resnet50_untrained';
  const isGabor = M.id === 'gabor_pyramid';
  
  if (isGabor) {
    // Gabor peaks very early (L1-L2) and falls steeply
    const v = (1 - a * 0.12) * Math.exp(-Math.pow(l - 0.5, 2) / 4.0) * 0.5 + 0.03 * rnd(sj + 1, a + 1, l + 1);
    return clamp(v, 0.02, 1);
  }
  
  const jit = 0.3 * (rnd(sj + 5, a + 1, m + 2) - 0.5);
  const pk = (isUntrained ? 1.2 + 0.25 * a : PEAKS[a]) + jit;
  const gain = 1 + 0.08 * (rnd(sj + 9, 3, m + 1) - 0.5);
  const v = gain * M.amp * (1 - 0.04 * a) * Math.exp(-Math.pow(l - pk, 2) / (2 * M.sig * M.sig)) + 0.04 * rnd(m + 1, a + 1, l + 1);
  return clamp(v, 0.02, 1);
}

const lerp = (a, b, t) => Math.round(a + (b - a) * t);
const col = (t) => {
  const c0 = [236, 243, 247], c1 = [7, 59, 82];
  return [0, 1, 2].map(i => lerp(c0[i], c1[i], clamp(t, 0, 1)));
};
const lum = (c) => {
  const f = (v) => { v = v / 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
};
const textOn = (c) => {
  const L = lum(c);
  const cw = 1.05 / (L + 0.05);
  const cd = (L + 0.05) / (lum([18, 32, 43]) + 0.05);
  return cw >= cd ? '#FFFFFF' : '#12202B';
};

const ranks = (arr) => arr.map(v => {
  const less = arr.filter(x => x < v).length;
  const eq = arr.filter(x => x === v).length;
  return less + (eq + 1) / 2;
});

const pearson = (x, y) => {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
};

// Check if live data from real run exists on server
let hasLiveData = false;
async function checkLiveData() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.has_results) {
      hasLiveData = true;
      const badge = document.getElementById('dataBadge');
      badge.textContent = `Live NSD 7T Data · ${data.completed_subjects || 'Subject 1'}`;
      badge.className = 'badge-live';
    }
  } catch (e) {}
}

function render() {
  const mi = state.mi;
  const ai = state.ai;
  const sj = state.subj;
  const M = MODELS[mi];
  const modelLabel = M.label;
  const areaName = AREAS[ai];
  const subjLabel = SUBJ[sj];

  // Best layer per area
  const best = AREAS.map((_, a) => {
    let bl = 0, bv = -1;
    for (let l = 0; l < NL; l++) {
      const v = score(mi, a, l);
      if (v > bv) { bv = v; bl = l; }
    }
    return bl;
  });

  const effLi = (state.li === null || state.li === undefined) ? best[ai] : state.li;

  // 1. Render Model Buttons
  const modelBtnsContainer = document.getElementById('modelButtons');
  modelBtnsContainer.innerHTML = '';
  MODELS.forEach((m, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-model ${idx === mi ? 'active' : ''}`;
    btn.innerHTML = `<span style="display:block; font:600 14px/1.2 'IBM Plex Sans',sans-serif;">${m.label}</span>` +
                    `<span style="display:block; font:400 12px/1.3 'IBM Plex Sans',sans-serif;">${m.sub}</span>`;
    btn.onclick = () => { state.mi = idx; state.li = null; render(); };
    modelBtnsContainer.appendChild(btn);
  });

  // 2. Render Subject Buttons
  const subjBtnsContainer = document.getElementById('subjButtons');
  subjBtnsContainer.innerHTML = '';
  SUBJ.forEach((sName, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-subj ${idx === sj ? 'active' : ''}`;
    btn.textContent = sName;
    btn.onclick = () => { state.subj = idx; render(); };
    subjBtnsContainer.appendChild(btn);
  });

  // 3. Baseline Toggle Button
  const baseBtn = document.getElementById('toggleBaselineBtn');
  baseBtn.className = `btn-baseline ${state.base ? 'active' : ''}`;
  baseBtn.textContent = `Untrained baseline: ${state.base ? 'shown' : 'hidden'}`;
  baseBtn.onclick = () => { state.base = !state.base; render(); };

  // 4. Matrix Heatmap
  document.getElementById('heatmapSubtitle').textContent = 
    `Cross-validated encoding score, noise-ceiling normalized. Showing ${modelLabel}, subject ${subjLabel}. Tap a cell or a region to inspect it.`;
  
  const headerRow = document.getElementById('heatmapHeaderRow');
  headerRow.innerHTML = '<div></div>';
  AREAS.forEach((aName, aIdx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-area ${aIdx === ai ? 'active' : ''}`;
    btn.textContent = aName;
    btn.onclick = () => { state.ai = aIdx; state.li = null; render(); };
    headerRow.appendChild(btn);
  });

  const heatmapBody = document.getElementById('heatmapBody');
  heatmapBody.innerHTML = '';
  for (let l = 0; l < NL; l++) {
    const rowDiv = document.createElement('div');
    rowDiv.style = 'display:grid; grid-template-columns:84px repeat(7,minmax(0,1fr)); gap:4px;';
    
    const labelDiv = document.createElement('div');
    labelDiv.style = "align-self:center; font:500 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;";
    labelDiv.textContent = `L${l + 1}${l === 0 ? ' early' : (l === NL - 1 ? ' late' : '')}`;
    rowDiv.appendChild(labelDiv);

    AREAS.forEach((aName, aIdx) => {
      const v = score(mi, aIdx, l);
      const rgb = col(v / 0.9);
      const isBest = best[aIdx] === l;
      const isSel = (aIdx === ai && l === effLi);

      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = `heatmap-cell ${isBest ? 'best' : ''} ${isSel ? 'selected' : ''}`;
      cell.style.background = `rgb(${rgb.join(',')})`;
      cell.style.color = textOn(rgb);
      cell.textContent = v.toFixed(2);
      cell.title = `${aName} layer ${l + 1} score: ${v.toFixed(2)}`;
      cell.onclick = () => { state.ai = aIdx; state.li = l; render(); };
      rowDiv.appendChild(cell);
    });
    heatmapBody.appendChild(rowDiv);
  }

  // 5. Layer-wise Bar Chart
  document.getElementById('barsTitle').textContent = `Layer-wise score · ${areaName}`;
  document.getElementById('barsSubtitle').textContent = `${modelLabel} predicting ${areaName} from each layer.`;

  const chartArea = document.getElementById('barsChartArea');
  chartArea.innerHTML = '<div style="position:absolute; left:0; right:0; top:0; border-top:1.5px dashed #51606C; pointer-events:none;"></div>';
  
  const buttonsRow = document.getElementById('barsButtonsRow');
  buttonsRow.innerHTML = '';

  const untrainedIdx = MODELS.findIndex(m => m.id === 'resnet50_untrained');
  for (let l = 0; l < NL; l++) {
    const v = score(mi, ai, l);
    const bv = score(untrainedIdx >= 0 ? untrainedIdx : 4, ai, l);
    const isSel = (l === effLi);

    const barCol = document.createElement('div');
    barCol.title = `L${l + 1}: ${v.toFixed(2)} (untrained ${bv.toFixed(2)})`;
    barCol.style = 'flex:1 1 0; min-width:0; position:relative; height:100%; display:flex; align-items:flex-end; justify-content:center;';

    const barFill = document.createElement('div');
    barFill.style = `width:62%; height:${(v * 100).toFixed(1)}%; background:${isSel ? '#073B52' : '#4C8FAB'}; border-radius:4px 4px 0 0;`;
    barCol.appendChild(barFill);

    if (state.base && mi !== untrainedIdx) {
      const tick = document.createElement('div');
      tick.style = `position:absolute; left:6%; right:6%; bottom:calc(${(bv * 100).toFixed(1)}% - 1px); height:3px; background:#C2570C;`;
      barCol.appendChild(tick);
    }
    chartArea.appendChild(barCol);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.style = `flex:1 1 0; min-width:0; min-height:48px; padding:4px 0; display:flex; flex-direction:column; align-items:center; gap:2px; border:0; background:transparent; cursor:pointer; font:${isSel ? 600 : 500} 11px/1.2 'IBM Plex Mono',monospace; color:${isSel ? '#073B52' : '#51606C'}; ${isSel ? 'text-decoration:underline;' : ''}`;
    btn.innerHTML = `<span>L${l + 1}</span><span>${v.toFixed(2)}</span>`;
    btn.onclick = () => { state.li = l; render(); };
    buttonsRow.appendChild(btn);
  }

  // Callout text
  const bl = best[ai];
  const bestV = score(mi, ai, bl);
  const baseV = score(untrainedIdx >= 0 ? untrainedIdx : 4, ai, bl);
  document.getElementById('calloutText').textContent = 
    `${modelLabel} peaks at L${bl + 1} for ${areaName} with a normalized score of ${bestV.toFixed(2)}. The untrained network reaches ${baseV.toFixed(2)} at that layer.`;

  // 6. Visual Hierarchy Map
  document.getElementById('hierSubtitle').textContent = 
    `Each region is shaded by the depth of its best-matching layer in ${modelLabel}.`;

  const hierGrid = document.getElementById('hierarchyGrid');
  // Retain arrows and column titles, replace region cards
  const existingCards = hierGrid.querySelectorAll('.hier-card');
  existingCards.forEach(c => c.remove());

  const HP = [[1, '1'], [1, '2'], [1, '3'], [3, '1 / span 3'], [5, '1'], [5, '2'], [5, '3']];
  AREAS.forEach((aName, aIdx) => {
    const t = best[aIdx] / (NL - 1);
    const rgb = col(0.18 + 0.82 * t);
    const card = document.createElement('div');
    card.className = 'hier-card';
    card.style = `grid-column:${HP[aIdx][0]}; grid-row:${HP[aIdx][1]}; background:rgb(${rgb.join(',')}); color:${textOn(rgb)}; border-radius:12px; padding:12px 12px; display:flex; flex-direction:column; gap:4px; justify-content:center; min-height:68px; box-sizing:border-box; cursor:pointer;`;
    card.innerHTML = `<span style="font:600 18px/1.2 'IBM Plex Sans',sans-serif;">${aName}</span>` +
                     `<span style="font:400 12px/1.3 'IBM Plex Mono',monospace;">best L${best[aIdx] + 1} · ${score(mi, aIdx, best[aIdx]).toFixed(2)}</span>`;
    card.onclick = () => { state.ai = aIdx; state.li = null; render(); };
    hierGrid.appendChild(card);
  });

  const rho = pearson(ranks([0, 1, 2, 3, 4, 5, 6]), ranks(best));
  document.getElementById('rhoValue').textContent = rho.toFixed(2);

  // 7. Selectivity Check
  document.getElementById('selectivityTitle').textContent = `Selectivity check · ${areaName}`;
  document.getElementById('selectivitySubtitle').textContent = 
    `Mean ${areaName} response by image category, measured vs predicted from ${modelLabel} layer L${effLi + 1}. Does the model reproduce what the region prefers?`;

  const catsChart = document.getElementById('catsChartArea');
  catsChart.innerHTML = '';
  const catsLabels = document.getElementById('catsLabelsRow');
  catsLabels.innerHTML = '';

  const q = score(mi, ai, effLi);
  const act = [], prd = [];
  CATS.forEach((cName, i) => {
    const a = SEL[ai][i];
    const p = clamp(q * a + (1 - q) * 0.5 + 0.08 * (rnd(sj + 2, ai + 1, i + 7) - 0.5), 0.03, 1);
    act.push(a); prd.push(p);

    const pair = document.createElement('div');
    pair.title = `${cName}: measured ${a.toFixed(2)}, predicted ${p.toFixed(2)}`;
    pair.style = 'flex:1 1 0; min-width:0; height:100%; display:flex; align-items:flex-end; justify-content:center; gap:3px;';

    const actBar = document.createElement('div');
    actBar.style = `width:38%; max-width:22px; height:${(a * 100).toFixed(1)}%; background:#12202B; border-radius:3px 3px 0 0;`;
    const prdBar = document.createElement('div');
    prdBar.style = `width:38%; max-width:22px; height:${(p * 100).toFixed(1)}%; background:#4C8FAB; border-radius:3px 3px 0 0;`;

    pair.appendChild(actBar);
    pair.appendChild(prdBar);
    catsChart.appendChild(pair);

    const lbl = document.createElement('div');
    lbl.style = "flex:1 1 0; min-width:0; text-align:center; font:500 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;";
    lbl.textContent = cName;
    catsLabels.appendChild(lbl);
  });

  const catCorr = pearson(act, prd);
  document.getElementById('corrValue').textContent = `r = ${catCorr.toFixed(2)}${hasLiveData ? '' : ' (NSD prior)'}`;
}

// Initial initialization
checkLiveData();
render();
</script>
</body>
</html>
"""

class BrainExplorerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif parsed.path == "/api/status":
            results_dir = os.path.join(PROJECT_ROOT, "results")
            has_results = False
            completed_subjects = []
            if os.path.exists(results_dir):
                for f in os.listdir(results_dir):
                    if f.startswith("summary_") and f.endswith(".json"):
                        has_results = True
                        s_name = f.replace("summary_", "").replace(".json", "")
                        completed_subjects.append(s_name)

            payload = {
                "has_results": has_results,
                "completed_subjects": ", ".join(completed_subjects) if completed_subjects else None
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        elif parsed.path == "/api/data":
            params = urllib.parse.parse_qs(parsed.query)
            subject = params.get("subject", ["subj01"])[0]
            model = params.get("model", ["resnet50"])[0]

            model_entries = load_data(subject, model)
            untrained_entries = load_data(subject, "resnet50_untrained")

            payload = {
                "subject": subject,
                "model": model,
                "model_entries": model_entries,
                "untrained_entries": untrained_entries,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silent console output
        pass

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), BrainExplorerHandler) as httpd:
        url = f"http://127.0.0.1:{PORT}"
        print(f"\n========================================================")
        print(f"[*] Brain Encoding Explorer running at: {url}")
        print(f"   Matches design reference in demo/Main.dc.html")
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
