/**
 * Brain Encoding Explorer - Client Application Logic
 * Fetches real computed results from /api/data and renders the interactive matrix.
 */

const state = {
  mi: 0,           // model index
  ai: 0,           // area index
  li: null,        // layer index
  subj: 'subj01',  // subject id
  data: {}         // holds REAL_DATA loaded from server
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
  document.getElementById('viewFigures').style.display = isExp ? 'none' : 'grid';
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
  if (!state.data || !state.data[subjId]) return [];
  const subjObj = state.data[subjId];
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

  // 1. Data Badge Status
  const badge = document.getElementById('dataBadge');
  if (hasData) {
    badge.textContent = `Real Data · ${subjId} (${entries.length} layers)`;
    badge.style.background = '#059669';
  } else {
    badge.textContent = `Awaiting Data for ${M.label}`;
    badge.style.background = 'rgba(255,255,255,0.2)';
  }

  // 2. Model Buttons
  const modelBtnsContainer = document.getElementById('modelButtons');
  modelBtnsContainer.innerHTML = '';
  MODELS.forEach((m, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-model ${idx === mi ? 'active' : ''}`;
    btn.innerHTML = `<span class="model-title">${m.label}</span>` +
                    `<span class="model-sub">${m.sub}</span>`;
    btn.onclick = () => { state.mi = idx; state.li = null; render(); };
    modelBtnsContainer.appendChild(btn);
  });

  // 3. Hierarchy Score Pill
  const hBox = document.getElementById('hierarchyScoreBox');
  const hMet = HIERARCHY_METRICS[M.id];
  if (hMet && hMet.rho !== null) {
    const isPos = hMet.rho > 0;
    hBox.innerHTML = `<span>Hierarchy ρ: <b style="color:${isPos ? '#059669':'#DC2626'}">${hMet.rho > 0 ? '+':''}${hMet.rho.toFixed(3)}</b> (p=${hMet.p.toFixed(4)}) · CoM: <b>${hMet.com_rho.toFixed(3)}</b></span>`;
  } else {
    hBox.innerHTML = `<span>Hierarchy: Baseline Control</span>`;
  }

  // 4. Global Champion Pill for Selected Area
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
                          `<span style="font:500 12px/1 'IBM Plex Mono',monospace; color:#64748B;">Subject 01 · Held-Out Test Set</span>`;

  // 5. Matrix Header Row
  const headerRow = document.getElementById('matrixHeaderRow');
  headerRow.innerHTML = '<div class="matrix-layer-label" style="background:#F8FAFC; border-radius:6px; padding:8px; font-weight:700;">Layer / Stage</div>';
  AREAS.forEach((aName, aIdx) => {
    const th = document.createElement('div');
    th.className = 'matrix-cell-th';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn-area ${aIdx === ai ? 'active' : ''}`;
    btn.textContent = aName;
    btn.onclick = () => { state.ai = aIdx; render(); };
    th.appendChild(btn);
    headerRow.appendChild(th);
  });

  // 6. Matrix Body Rows
  const tableBody = document.getElementById('matrixTableBody');
  tableBody.innerHTML = '';

  const bestPerArea = {};
  AREAS.forEach(aName => {
    let maxVal = -1, bestL = -1;
    entries.forEach((e, lIdx) => {
      const val = (aName === 'Overall') ? e.overall_median_r : (e.roi_medians ? e.roi_medians[aName] : null);
      if (val && val > maxVal) { maxVal = val; bestL = lIdx; }
    });
    bestPerArea[aName] = bestL;
  });

  if (!hasData) {
    const emptyRow = document.createElement('div');
    emptyRow.style = 'display:table-row;';
    emptyRow.innerHTML = `<div style="display:table-cell; padding:32px; text-align:center; color:#94A3B8; font:500 14px/1.4 'IBM Plex Sans',sans-serif; background:#F8FAFC; border-radius:10px; border:1px dashed #CBD5E1;" colspan="12">
      No computed layer representations currently loaded for ${M.label}.
    </div>`;
    tableBody.appendChild(emptyRow);
  } else {
    entries.forEach((entry, lIdx) => {
      const row = document.createElement('div');
      row.className = 'matrix-row';

      const label = document.createElement('div');
      label.className = 'matrix-layer-label';
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

      tableBody.appendChild(row);
    });
  }

  // 7. Detail Representation Panel
  const activeLayerIdx = (state.li !== null && state.li < entries.length) 
    ? state.li 
    : (entries.length > 0 ? (bestPerArea[areaName] ?? 0) : null);
  const activeLayer = (activeLayerIdx !== null && entries[activeLayerIdx]) ? entries[activeLayerIdx] : null;

  if (activeLayer) {
    document.getElementById('detailLayerName').textContent = activeLayer.layer_name;
    document.getElementById('detailLayerDepth').textContent = (activeLayer.normalized_depth || 0).toFixed(3);
    const scoreVal = (areaName === 'Overall') 
      ? activeLayer.overall_median_r 
      : (activeLayer.roi_medians ? activeLayer.roi_medians[areaName] : null);
    document.getElementById('detailScore').textContent = scoreVal ? `r = ${scoreVal.toFixed(4)}` : 'r = —';
    document.getElementById('detailHeader').textContent = `${M.label} · ${activeLayer.layer_desc || activeLayer.layer_name}`;
  } else {
    document.getElementById('detailLayerName').textContent = '—';
    document.getElementById('detailLayerDepth').textContent = '—';
    document.getElementById('detailScore').textContent = '—';
  }

  // 8. Populate ROI Profile Bars
  const profileContainer = document.getElementById('roiProfileBars');
  profileContainer.innerHTML = '';
  const roisOnly = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'OPA', 'EBA', 'FFA', 'PPA', 'RSC'];
  
  roisOnly.forEach(roi => {
    const rRow = document.createElement('div');
    rRow.className = 'profile-bar-row';

    const rLbl = document.createElement('span');
    rLbl.className = 'profile-bar-roi';
    rLbl.textContent = roi;
    rRow.appendChild(rLbl);

    const barTrack = document.createElement('div');
    barTrack.className = 'profile-bar-track';

    const val = (activeLayer && activeLayer.roi_medians && typeof activeLayer.roi_medians[roi] === 'number')
      ? activeLayer.roi_medians[roi]
      : 0;

    const barFill = document.createElement('div');
    const pct = Math.max(0, Math.min(100, (val / 0.65) * 100));
    barFill.className = 'profile-bar-fill';
    barFill.style.width = `${pct}%`;
    barFill.style.background = (roi === areaName) ? '#EA580C' : '#0B6E99';
    barTrack.appendChild(barFill);
    rRow.appendChild(barTrack);

    const valLbl = document.createElement('span');
    valLbl.className = 'profile-bar-num';
    valLbl.textContent = val > 0 ? val.toFixed(3) : '—';
    rRow.appendChild(valLbl);

    profileContainer.appendChild(rRow);
  });
}

// Initial Data Fetch
async function init() {
  try {
    const res = await fetch('/api/data');
    if (res.ok) {
      state.data = await res.json();
    }
  } catch (err) {
    console.warn("Could not fetch /api/data, using fallback if provided.", err);
  }
  render();
}

document.addEventListener('DOMContentLoaded', init);
