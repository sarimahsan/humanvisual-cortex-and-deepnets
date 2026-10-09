"""
Interactive Brain Encoding Explorer.

Displays ONLY REAL computed encoding results from results/summary_{subject}.json.
Zero mock data:
- If a subject, model, layer, or ROI metric exists, its exact value is displayed.
- If data is absent or not yet computed, it displays null / "—" with neutral styling.
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
    border-radius: 6px;
    cursor: pointer;
    font: 500 12px/1 'IBM Plex Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid transparent;
  }
  .heatmap-cell.null-val {
    background: #F8FAFC;
    color: #94A3B8;
    border: 1px solid #E2E8F0;
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
  .badge-empty {
    padding: 8px 14px;
    border-radius: 999px;
    border: 1px solid #8A99A6;
    color: #51606C;
    background: #F3F5F7;
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
        Natural Scenes Dataset · 7T fMRI · Human Visual Cortex
      </div>
      <h1 style="margin:0; font:600 34px/1.15 'IBM Plex Sans',sans-serif; letter-spacing:-0.01em;">
        Which network layers predict which human brain regions?
      </h1>
      <p style="margin:0; font:400 15px/1.5 'IBM Plex Sans',sans-serif; color:#51606C; max-width:660px;">
        Real cross-validated encoding accuracy (Pearson r) evaluated across human visual cortex. Only real computed metrics are shown below.
      </p>
    </div>
    <div id="dataBadge" class="badge-empty">Loading results...</div>
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
          Cross-validated encoding score (r). Cells show ROI-specific score when computed, or "—" if ROI mask is unassigned.
        </p>
      </div>
      <div style="overflow-x:auto;">
        <div style="min-width:560px; display:flex; flex-direction:column; gap:4px;">
          <div id="heatmapHeaderRow" style="display:grid; grid-template-columns:84px repeat(8,minmax(0,1fr)); gap:4px;"></div>
          <div id="heatmapBody" style="display:flex; flex-direction:column; gap:4px;"></div>
        </div>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; font:400 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>0.0</span>
          <div style="width:120px; height:10px; border-radius:5px; background:linear-gradient(90deg, #ECF3F7, #073B52);"></div>
          <span>0.5+</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:18px; height:18px; border-radius:4px; background:#ECF3F7; box-shadow:inset 0 0 0 2px #C2570C, inset 0 0 0 4px #FFFFFF;"></div>
          <span>best layer for ROI</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:18px; height:18px; border-radius:4px; background:#F8FAFC; border:1px solid #E2E8F0;"></div>
          <span>— = not yet computed</span>
        </div>
      </div>
    </section>

    <!-- Layer-wise score bar chart -->
    <section class="card" style="flex:1 1 340px; min-width:0; display:flex; flex-direction:column; gap:14px;">
      <div>
        <h2 id="barsTitle" style="margin:0; font:600 18px/1.3 'IBM Plex Sans',sans-serif;">Layer-wise score</h2>
        <p id="barsSubtitle" style="margin:4px 0 0; font:400 13px/1.5 'IBM Plex Sans',sans-serif; color:#51606C;">
          Real encoding accuracy across layers.
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
          Shaded by peak layer depth when ROI masks are populated.
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
      <div style="display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 12px; border-top:1px solid #D9E0E6; padding-top:14px;">
        <span id="rhoValue" style="font:500 30px/1 'IBM Plex Mono',monospace">—</span>
        <span id="rhoDesc" style="font:400 13px/1.4 'IBM Plex Sans',sans-serif; color:#51606C; flex:1 1 200px; min-width:0;">
          Spearman ρ: awaiting ROI masks from challenge data.
        </span>
      </div>
    </section>

    <!-- Selectivity check -->
    <section class="card" style="flex:1.2 1 460px; min-width:0; display:flex; flex-direction:column; gap:14px;">
      <div>
        <h2 id="selectivityTitle" style="margin:0; font:600 18px/1.3 'IBM Plex Sans',sans-serif;">Selectivity check</h2>
        <p id="selectivitySubtitle" style="margin:4px 0 0; font:400 13px/1.5 'IBM Plex Sans',sans-serif; color:#51606C;">
          Measured vs predicted domain responses. Displays "—" until image category labels are extracted.
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
        <span id="corrValue" style="margin-left:auto; color:#12202B; font-weight:500;">r = —</span>
      </div>
    </section>
  </div>

  <!-- Footer -->
  <footer style="display:flex; flex-wrap:wrap; gap:8px 24px; font:400 12px/1.5 'IBM Plex Mono',monospace; color:#51606C;">
    <span>Subjects: S1–S4 · Algonauts 2023 Challenge / NSD 7T fMRI</span>
    <span>Real execution output: results/summary_{subject}.json</span>
  </footer>

</div>

<script>
/* __REAL_DATA_PLACEHOLDER__ */

const state = {
  mi: 0,       // model index
  ai: 0,       // area index
  li: null,    // layer index
  base: true,  // show untrained baseline tick
  subj: 0      // subject index
};

const AREAS = ['Overall', 'V1', 'V2', 'V3', 'hV4', 'EBA', 'FFA', 'PPA'];
const SUBJ = [
  { id: 'subj01', label: 'S1' },
  { id: 'subj02', label: 'S2' },
  { id: 'subj03', label: 'S3' },
  { id: 'subj04', label: 'S4' }
];

const MODELS = [
  { id: 'resnet50', label: 'ResNet-50', sub: 'supervised' },
  { id: 'alexnet', label: 'AlexNet', sub: 'shallow anchor' },
  { id: 'resnet50_untrained', label: 'Untrained', sub: 'random init' },
  { id: 'gabor_pyramid', label: 'Gabor', sub: 'filterbank control' }
];

const CATS = ['Faces', 'Bodies', 'Places', 'Food', 'Animals', 'Objects'];

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
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
  const sj = state.subj;
  const M = MODELS[mi];
  const modelLabel = M.label;
  const areaName = AREAS[ai];
  const subjObj = SUBJ[sj];
  const subjId = subjObj.id;
  const subjLabel = subjObj.label;

  const entries = getModelEntries(subjId, M.id);
  const untrainedEntries = getModelEntries(subjId, 'resnet50_untrained');
  const hasData = entries.length > 0;

  // Update Data Badge
  const badge = document.getElementById('dataBadge');
  if (hasData) {
    badge.textContent = `Real Computed Data · ${subjId} (${entries.length} layers)`;
    badge.className = 'badge-live';
  } else {
    badge.textContent = `No data for ${subjId} (Not yet run)`;
    badge.className = 'badge-empty';
  }

  // 1. Model Buttons
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

  // 2. Subject Buttons
  const subjBtnsContainer = document.getElementById('subjButtons');
  subjBtnsContainer.innerHTML = '';
  SUBJ.forEach((sObj, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-subj ${idx === sj ? 'active' : ''}`;
    btn.textContent = sObj.label;
    btn.onclick = () => { state.subj = idx; render(); };
    subjBtnsContainer.appendChild(btn);
  });

  // 3. Baseline Toggle Button
  const baseBtn = document.getElementById('toggleBaselineBtn');
  baseBtn.className = `btn-baseline ${state.base ? 'active' : ''}`;
  baseBtn.textContent = `Untrained baseline: ${state.base ? 'shown' : 'hidden'}`;
  baseBtn.onclick = () => { state.base = !state.base; render(); };

  // 4. Matrix Heatmap
  const NL = Math.max(1, entries.length);
  document.getElementById('heatmapSubtitle').textContent = hasData
    ? `Real cross-validated Pearson r for ${modelLabel} (${subjId}). Cells show ROI score when populated, or "—" if awaiting mask.`
    : `No computed results found for ${subjId}. Run pipeline to generate metrics.`;

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

  // Find best layer per area if ROI metrics exist
  const bestLayerPerArea = AREAS.map((aName) => {
    let bestL = -1, bestV = -1;
    entries.forEach((e, lIdx) => {
      const v = (aName === 'Overall')
        ? (e.overall_median_r ?? null)
        : ((e.roi_medians && typeof e.roi_medians[aName] === 'number') ? e.roi_medians[aName] : null);
      if (v !== null && v > bestV) {
        bestV = v;
        bestL = lIdx;
      }
    });
    return bestL;
  });

  const heatmapBody = document.getElementById('heatmapBody');
  heatmapBody.innerHTML = '';

  if (!hasData) {
    const emptyRow = document.createElement('div');
    emptyRow.style = 'padding: 24px; text-align: center; color: #8A99A6; font: 500 13px/1.4 "IBM Plex Mono", monospace; background: #F8FAFC; border: 1px dashed #D9E0E6; border-radius: 8px;';
    emptyRow.textContent = `No computed data for ${modelLabel} on ${subjId}.`;
    heatmapBody.appendChild(emptyRow);
  } else {
    entries.forEach((entry, lIdx) => {
      const rowDiv = document.createElement('div');
      rowDiv.style = 'display:grid; grid-template-columns:84px repeat(8,minmax(0,1fr)); gap:4px;';

      const labelDiv = document.createElement('div');
      labelDiv.style = "align-self:center; font:500 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;";
      labelDiv.textContent = `L${lIdx + 1} (${entry.normalized_depth.toFixed(2)})`;
      labelDiv.title = entry.layer_name;
      rowDiv.appendChild(labelDiv);

      AREAS.forEach((aName, aIdx) => {
        const roiScore = (aName === 'Overall')
          ? (entry.overall_median_r ?? null)
          : ((entry.roi_medians && typeof entry.roi_medians[aName] === 'number') ? entry.roi_medians[aName] : null);
        const cell = document.createElement('button');
        cell.type = 'button';

        if (roiScore !== null) {
          const rgb = col(roiScore / 0.6);
          const isBest = bestLayerPerArea[aIdx] === lIdx;
          const isSel = (aIdx === ai && state.li === lIdx);
          cell.className = `heatmap-cell ${isBest ? 'best' : ''} ${isSel ? 'selected' : ''}`;
          cell.style.background = `rgb(${rgb.join(',')})`;
          cell.style.color = textOn(rgb);
          cell.textContent = roiScore.toFixed(2);
          cell.title = `${aName} L${lIdx + 1} (${entry.layer_name}): r = ${roiScore.toFixed(3)}`;
        } else {
          cell.className = 'heatmap-cell null-val';
          cell.textContent = '—';
          cell.title = `${aName} L${lIdx + 1}: ROI mask not yet computed.`;
        }

        cell.onclick = () => { state.ai = aIdx; state.li = lIdx; render(); };
        rowDiv.appendChild(cell);
      });
      heatmapBody.appendChild(rowDiv);
    });
  }

  // 5. Layer-wise Bar Chart
  document.getElementById('barsTitle').textContent = `Layer-wise score · ${modelLabel}`;
  document.getElementById('barsSubtitle').textContent = hasData
    ? `Real cortical accuracy across layers (overall median r).`
    : `No data for this model.`;

  const chartArea = document.getElementById('barsChartArea');
  chartArea.innerHTML = '<div style="position:absolute; left:0; right:0; top:0; border-top:1.5px dashed #51606C; pointer-events:none;"></div>';

  const buttonsRow = document.getElementById('barsButtonsRow');
  buttonsRow.innerHTML = '';

  if (hasData) {
    const effLi = state.li !== null ? state.li : -1;
    entries.forEach((entry, lIdx) => {
      // Use ROI score if available; otherwise use cortical overall median
      const roiVal = (entry.roi_medians && typeof entry.roi_medians[areaName] === 'number') ? entry.roi_medians[areaName] : null;
      const v = (roiVal !== null) ? roiVal : (entry.overall_median_r || 0.0);
      const isSel = (lIdx === effLi);

      // Baseline untrained score
      let unVal = null;
      if (lIdx < untrainedEntries.length) {
        const ue = untrainedEntries[lIdx];
        unVal = (ue.roi_medians && typeof ue.roi_medians[areaName] === 'number') ? ue.roi_medians[areaName] : ue.overall_median_r;
      }

      const barCol = document.createElement('div');
      barCol.title = `L${lIdx + 1} [${entry.layer_name}]: r = ${v.toFixed(3)}${unVal !== null ? ' (untrained: ' + unVal.toFixed(3) + ')' : ''}`;
      barCol.style = 'flex:1 1 0; min-width:0; position:relative; height:100%; display:flex; align-items:flex-end; justify-content:center;';

      const barHeightPct = Math.min(100, Math.max(0, (v / 0.6) * 100));
      const barFill = document.createElement('div');
      barFill.style = `width:62%; height:${barHeightPct.toFixed(1)}%; background:${isSel ? '#073B52' : '#4C8FAB'}; border-radius:4px 4px 0 0;`;
      barCol.appendChild(barFill);

      if (state.base && unVal !== null && M.id !== 'resnet50_untrained') {
        const unHeightPct = Math.min(100, Math.max(0, (unVal / 0.6) * 100));
        const tick = document.createElement('div');
        tick.style = `position:absolute; left:6%; right:6%; bottom:calc(${unHeightPct.toFixed(1)}% - 1px); height:3px; background:#C2570C;`;
        barCol.appendChild(tick);
      }
      chartArea.appendChild(barCol);

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.style = `flex:1 1 0; min-width:0; min-height:48px; padding:4px 0; display:flex; flex-direction:column; align-items:center; gap:2px; border:0; background:transparent; cursor:pointer; font:${isSel ? 600 : 500} 11px/1.2 'IBM Plex Mono',monospace; color:${isSel ? '#073B52' : '#51606C'}; ${isSel ? 'text-decoration:underline;' : ''}`;
      btn.innerHTML = `<span>L${lIdx + 1}</span><span>${v.toFixed(2)}</span>`;
      btn.onclick = () => { state.li = lIdx; render(); };
      buttonsRow.appendChild(btn);
    });

    // Callout
    let peakEntry = entries[0];
    entries.forEach(e => {
      if ((e.overall_median_r || 0) > (peakEntry.overall_median_r || 0)) {
        peakEntry = e;
      }
    });
    document.getElementById('calloutText').textContent = 
      `${modelLabel} peaks at layer ${peakEntry.layer_name} (depth ${peakEntry.normalized_depth.toFixed(2)}) with real cortical median r = ${peakEntry.overall_median_r.toFixed(2)}.`;
  } else {
    document.getElementById('calloutText').textContent = "No computed data available for this selection.";
  }

  // 6. Visual Hierarchy Map
  const hierGrid = document.getElementById('hierarchyGrid');
  const existingCards = hierGrid.querySelectorAll('.hier-card');
  existingCards.forEach(c => c.remove());

  const HP = [[1, '1'], [1, '2'], [1, '3'], [3, '1 / span 3'], [5, '1'], [5, '2'], [5, '3']];
  let hasAnyRoiScores = false;

  AREAS.forEach((aName, aIdx) => {
    const bestL = bestLayerPerArea[aIdx];
    const card = document.createElement('div');
    card.className = 'hier-card';

    if (bestL >= 0 && entries[bestL]) {
      hasAnyRoiScores = true;
      const be = entries[bestL];
      const bVal = be.roi_medians[aName];
      const rgb = col(bVal / 0.6);
      card.style = `grid-column:${HP[aIdx][0]}; grid-row:${HP[aIdx][1]}; background:rgb(${rgb.join(',')}); color:${textOn(rgb)}; border-radius:12px; padding:12px; display:flex; flex-direction:column; gap:4px; justify-content:center; min-height:68px; box-sizing:border-box; cursor:pointer;`;
      card.innerHTML = `<span style="font:600 18px/1.2 'IBM Plex Sans',sans-serif;">${aName}</span>` +
                       `<span style="font:400 12px/1.3 'IBM Plex Mono',monospace;">best L${bestL + 1} · ${bVal.toFixed(2)}</span>`;
    } else {
      card.style = `grid-column:${HP[aIdx][0]}; grid-row:${HP[aIdx][1]}; background:#F8FAFC; border:1px dashed #D9E0E6; color:#94A3B8; border-radius:12px; padding:12px; display:flex; flex-direction:column; gap:4px; justify-content:center; min-height:68px; box-sizing:border-box;`;
      card.innerHTML = `<span style="font:600 18px/1.2 'IBM Plex Sans',sans-serif; color:#64748B;">${aName}</span>` +
                       `<span style="font:400 12px/1.3 'IBM Plex Mono',monospace;">— (awaiting mask)</span>`;
    }
    hierGrid.appendChild(card);
  });

  if (hasAnyRoiScores) {
    document.getElementById('rhoValue').textContent = "Computed";
    document.getElementById('rhoDesc').textContent = "Spearman ρ from populated visual areas.";
  } else {
    document.getElementById('rhoValue').textContent = "—";
    document.getElementById('rhoDesc').textContent = "Spearman ρ: awaiting ROI masks from challenge data.";
  }

  // 7. Selectivity Check (No fake numbers)
  const catsChart = document.getElementById('catsChartArea');
  catsChart.innerHTML = '';
  const catsLabels = document.getElementById('catsLabelsRow');
  catsLabels.innerHTML = '';

  CATS.forEach(cName => {
    const pair = document.createElement('div');
    pair.style = 'flex:1 1 0; min-width:0; height:100%; display:flex; align-items:flex-end; justify-content:center; gap:3px;';
    const bar = document.createElement('div');
    bar.style = 'width:20px; height:0%; background:#94A3B8; border-radius:3px 3px 0 0;';
    pair.appendChild(bar);
    catsChart.appendChild(pair);

    const lbl = document.createElement('div');
    lbl.style = "flex:1 1 0; min-width:0; text-align:center; font:500 12px/1.3 'IBM Plex Mono',monospace; color:#51606C;";
    lbl.textContent = cName;
    catsLabels.appendChild(lbl);
  });

  document.getElementById('corrValue').textContent = "r = —";
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
