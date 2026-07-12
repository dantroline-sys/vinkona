"""A tiny, dependency-free browser for inspecting the knowledge base.

Tabs let you *peruse* (not just search) everything the pipeline produces, so you
can debug it:
  * **Search**       — the real kb_search (dense + sparse + rerank) with scores.
  * **Raw**          — random ingested chunks (spot OCR/boilerplate problems).
  * **Concepts**     — distilled nodes: the meaning + provenance.
  * **Relations**    — typed edges with their mechanism, regime, and any meta
                       links (disagrees_with / alternative_to / context_variant_of)
                       — the window into reconciliation.
  * **Cards**        — procedure cards (how).
  * **Sources**      — the source registry: trust + epistemic regime per doc.
  * **Adjudication** — the node-merge queue (ambiguous link_to_node matches).
  * **Gaps**         — queries the KB couldn't answer.

No CDN/asset dependencies; everything is inline.
"""
from __future__ import annotations

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Host — viewer</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0; }
  header { padding: 12px 20px 0; border-bottom: 1px solid #8884;
           position: sticky; top: 0; background: Canvas; z-index: 2; }
  h1 { font-size: 17px; margin: 0 0 6px; }
  #stats { font-size: 13px; opacity: .85; margin-bottom: 6px; }
  #live { display: flex; gap: 16px; flex-wrap: wrap; align-items: baseline;
          font-size: 13px; margin-bottom: 8px; }
  #live .stat { display: inline-flex; gap: 5px; align-items: baseline; }
  #live .lbl { opacity: .6; }
  #live .val { font-weight: 600; font-variant-numeric: tabular-nums; }
  #live .rate { font-size: 11px; font-variant-numeric: tabular-nums; }
  #live .rate.up { color: #22aa66; }
  #live .rate.zero { opacity: .35; }
  #live .upd { opacity: .5; font-size: 11px; margin-left: auto; }
  .badge { display: inline-block; padding: 1px 7px; margin: 0 6px 2px 0;
           border: 1px solid #8886; border-radius: 10px; font-size: 12px; }
  .tabs { display: flex; gap: 2px; flex-wrap: wrap; }
  .tabs button { font: inherit; font-size: 13px; padding: 6px 12px; cursor: pointer;
           border: 1px solid #8886; border-bottom: none; border-radius: 6px 6px 0 0;
           background: #8881; color: CanvasText; }
  .tabs button.active { background: Canvas; font-weight: 600; position: relative; top: 1px; }
  main { padding: 16px 20px; max-width: 1000px; }
  .bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
  input, select, button { font: inherit; padding: 7px 10px; border: 1px solid #8886;
           border-radius: 6px; background: Canvas; color: CanvasText; }
  input[type=search] { flex: 1; min-width: 240px; }
  .toolbtn { cursor: pointer; }
  .note { padding: 8px 12px; border-radius: 6px; margin: 10px 0;
          background: #f5a62322; border: 1px solid #f5a62366; font-size: 14px; }
  .p { border: 1px solid #8883; border-radius: 8px; padding: 10px 12px; margin: 10px 0; }
  .p .meta { font-size: 12px; opacity: .85; margin-bottom: 5px;
             display: flex; gap: 8px; flex-wrap: wrap; align-items: baseline; }
  .p .title { font-weight: 600; }
  .p .score { margin-left: auto; font-variant-numeric: tabular-nums; }
  .p .text { white-space: pre-wrap; }
  .p .src { font-size: 11px; opacity: .65; margin-top: 6px; word-break: break-all; }
  .rel { font-size: 15px; }
  .rel .arrow { font-weight: 600; }
  .meta-link { font-size: 12px; margin-top: 5px; }
  .meta-link.warn { color: #d9534f; }
  ol.steps { margin: 6px 0 0; padding-left: 22px; }
  .focus { border-left: 3px solid #4a90d9; padding: 6px 12px; margin: 4px 0 12px; }
  .focus .focus-l { font-size: 18px; font-weight: 600; }
  .focus .focus-s { opacity: .8; margin-top: 2px; }
  .crit { margin: 6px 0; display: flex; flex-direction: column; gap: 4px; }
  .crit .crit-row { font-size: 13px; }
  .crit .crit-row b { font-size: 11px; text-transform: uppercase; opacity: .6; margin-right: 6px; }
  .struct { margin-top: 14px; border-top: 1px solid #8883; padding-top: 8px; }
  .struct .struct-h { font-size: 11px; text-transform: uppercase; opacity: .55; margin-bottom: 6px; }
  .struct .struct-sec { margin: 5px 0; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
  .struct .struct-role { font-size: 12px; opacity: .7; min-width: 130px; font-weight: 600; }
  .struct .struct-ent { font-size: 13px; border: 1px solid #8883; border-radius: 6px; padding: 1px 8px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #8883; }
  th { opacity: .7; font-weight: 600; }
  .empty { opacity: .6; padding: 20px 0; }
  /* per-tab help (help.json via /help) + the Import formats table (live probes) */
  .tabhelp { margin: 0 0 12px; font-size: 13px; }
  .tabhelp > summary { cursor: pointer; opacity: .6; list-style: none; }
  .tabhelp > summary::-webkit-details-marker { display: none; }
  .tabhelp > div { opacity: .85; border-left: 2px solid #8886; padding: 8px 12px;
                   margin-top: 6px; max-width: 860px; }
  .fmt-ok { color: #22aa66; font-weight: 600; }
  .fmt-no { color: #d9534f; font-weight: 600; }
  #fmtbox { margin: 10px 0 14px; }
  #fmtbox .fmt-h { font-size: 12px; text-transform: uppercase; opacity: .55; margin-bottom: 4px; }
</style>
</head>
<body>
<header>
  <h1>Knowledge Host — viewer</h1>
  <div id="stats">loading…</div>
  <div id="live"></div>
  <div class="tabs" id="tabs"></div>
</header>
<main>
  <div class="bar" id="bar"></div>
  <div id="tabhelp"></div>
  <div id="banner"></div>
  <div id="results" class="empty">…</div>
</main>
<script>
const $ = s => document.querySelector(s);
const esc = t => { const d = document.createElement('div'); d.textContent = t == null ? '' : t; return d.innerHTML; };
const badge = t => `<span class="badge">${esc(t)}</span>`;

const TABS = [
  ['ask', 'Ask'], ['search', 'Search'], ['raw', 'Raw'], ['nodes', 'Concepts'], ['edges', 'Relations'],
  ['cards', 'Cards'], ['sources', 'Sources'], ['adjudication', 'Adjudication'], ['gaps', 'Gaps'],
  ['bundles', 'Bundles'], ['library', 'Library'], ['ops', 'Operations'],
  ['autopilot', 'Prioritizer'], ['settings', 'Settings'],
];
let active = 'ask';

function buildTabs() {
  $('#tabs').innerHTML = TABS.map(([k, lbl]) =>
    `<button data-k="${k}" onclick="go('${k}')">${lbl}</button>`).join('');
}
// ── Help: tab intros from help.json + live import-format probes (/help) ─────
let HELP = { help: {}, formats: [] };
async function loadHelp() {
  try { HELP = await (await fetch('/help')).json(); } catch (e) { /* keep empty */ }
  renderHelp(active);
}
function renderHelp(k) {
  const box = $('#tabhelp');
  const txt = (HELP.help || {})[k];
  let h = txt ? `<details class="tabhelp"><summary>ⓘ about this tab</summary><div>${esc(txt)}</div></details>` : '';
  if (k === 'ops' && (HELP.formats || []).length) {
    const rows = HELP.formats.map(f =>
      `<tr><td>${esc(f.format)}</td><td><code>${esc(f.matches)}</code></td>
       <td class="${f.ready ? 'fmt-ok' : 'fmt-no'}">${f.ready ? '✓ ready' : '✗ not installed'}</td>
       <td style="opacity:.75">${esc(f.how)}</td></tr>`).join('');
    h += `<details id="fmtbox"><summary style="cursor:pointer;opacity:.6;font-size:13px">📄 Import formats — what this host can ingest (click to expand)</summary>
      <div style="opacity:.85;font-size:13px;margin:8px 0;max-width:860px">${esc((HELP.help || {}).import || '')}</div>
      <table><tr><th>format</th><th>files</th><th>status</th><th>enable / notes</th></tr>${rows}</table></details>`;
  }
  box.innerHTML = h;
}

function go(k) {
  active = k;
  if (opsTimer) { clearInterval(opsTimer); opsTimer = null; }   // stop polling when leaving Ops
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.k === k));
  renderBar(k);
  renderHelp(k);
  if (k === 'ask') { $('#results').className = 'empty'; $('#results').textContent = 'Ask the structured KB a what / how / why question.'; $('#aq') && $('#aq').focus(); }
  else if (k === 'search') { $('#results').className = 'empty'; $('#results').textContent = 'Type a query above.'; $('#q') && $('#q').focus(); }
  else if (k === 'ops') { loadOps(); opsTimer = setInterval(() => { if (active === 'ops') pollOps(); }, 2500); }
  else if (k === 'settings') { loadSettings(); }
  else if (k === 'autopilot') { loadAutopilot(); }
  else if (k === 'bundles') { loadBundles(); }
  else if (k === 'library') { loadLibrary(); }
  else load(k);
}

function renderBar(k) {
  $('#banner').innerHTML = '';
  if (k === 'ask') {
    $('#bar').innerHTML =
      `<input id="aq" type="search" placeholder="Ask a what / how / why question…" autofocus
              onkeydown="if(event.key==='Enter')doAsk()">
       <select id="rigor" title="rigor"><option value="">auto rigor</option>
         <option value="low">low</option><option value="high">high (stakes)</option></select>
       <button class="toolbtn" onclick="doAsk()">Ask</button>`;
  } else if (k === 'search') {
    $('#bar').innerHTML =
      `<input id="q" type="search" placeholder="Search the knowledge base…" autofocus
              onkeydown="if(event.key==='Enter')doSearch()">
       <input id="k" type="number" value="8" min="1" max="50" style="width:70px" title="passages">
       <button class="toolbtn" onclick="doSearch()">Search</button>`;
  } else if (k === 'raw') {
    $('#bar').innerHTML =
      `<select id="srcfilter" title="source"></select>
       <button class="toolbtn" onclick="load('raw')">Reload sample</button>`;
    fillSources();
  } else if (k === 'ops') {
    $('#bar').innerHTML = tokInput()
      + ` <select id="opcmd" onchange="onCmdChange()" title="maintenance command"></select>`
      + ` <span id="opopts"></span>`
      + ` <button class="toolbtn" onclick="runOp()">Run</button>`
      + ` <button class="toolbtn" onclick="stopOp()">Stop</button>`
      + ` <button class="toolbtn" onclick="reloadKB()" title="re-warm caches after a job">Reload KB</button>`;
  } else if (k === 'bundles') {
    $('#bar').innerHTML = tokInput()
      + ` <select id="scensel" title="scenario"></select>`
      + ` <button class="toolbtn" onclick="applyScenario()" title="reassemble + hot-swap the live session">Apply scenario</button>`
      + ` <button class="toolbtn" onclick="loadBundles()">Refresh</button>`;
  } else if (k === 'autopilot') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="addAutopilotStep()">+ Add step</button>`
      + ` <button class="toolbtn" onclick="saveAutopilot()">Save plan</button>`
      + ` <button class="toolbtn" onclick="loadAutopilot()">Refresh</button>`;
  } else if (k === 'settings') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="saveSettings()">Save</button>`
      + ` <span style="opacity:.6;font-size:13px">edits config.toml — restart or Reload KB to apply</span>`;
  } else if (k === 'library') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="saveLibrarySelection()">Save selection</button>`
      + ` <button class="toolbtn" onclick="loadLibrary()">Refresh</button>`;
  } else {
    $('#bar').innerHTML = `<button class="toolbtn" onclick="load('${k}')">Reload</button>
       <span style="opacity:.6;font-size:13px">newest / random first</span>`;
  }
}

// ── live counters: poll every few seconds, show per-minute rates ──────────────
const REFRESH_MS = 3000, RATE_WINDOW_MS = 60000;
const fmt = n => (n || 0).toLocaleString();
let statHist = [];          // rolling [{t, c:{metric:value}}] over the last minute

function computeRates(counts) {
  const now = Date.now();
  statHist.push({ t: now, c: counts });
  while (statHist.length > 2 && now - statHist[0].t > RATE_WINDOW_MS) statHist.shift();
  const base = statHist[0], dt = (now - base.t) / 1000;   // seconds spanned
  const rates = {};
  for (const k in counts) rates[k] = dt >= 1 ? (counts[k] - (base.c[k] || 0)) / dt * 60 : null;
  return { dt, rates };
}
function rateSpan(r) {
  if (r == null) return '';
  const v = Math.abs(r) >= 10 ? Math.round(r) : Math.round(r * 10) / 10;
  return `<span class="rate ${r > 0.049 ? 'up' : 'zero'}">${v > 0 ? '+' : ''}${v}/min</span>`;
}
function statEl(lbl, val, r) {
  return `<span class="stat" title="${lbl}"><span class="lbl">${lbl}</span>`
    + `<span class="val">${fmt(val)}</span>${rateSpan(r)}</span>`;
}

async function refreshStats() {
  let s = {}, kb = {};
  try { s = await (await fetch('stats')).json(); } catch (e) {}
  try { kb = (await (await fetch('kb')).json()).counts || {}; } catch (e) {}
  const counts = {
    chunks: s.chunks || 0, nodes: kb.nodes || 0, edges: kb.edges || 0, cards: kb.cards || 0,
    distilled: kb.distilled_chunks || 0, adjudicate: kb.merge_candidates || 0, gaps: kb.gaps || 0,
  };
  const { dt, rates } = computeRates(counts);
  const by = Object.entries(s.by_source || {}).map(([k, v]) => badge(`${k}: ${fmt(v)}`)).join('');
  $('#stats').innerHTML = badge('backend: ' + (s.backend || '?'))
    + badge('dense: ' + (s.dense ? 'yes' : 'no')) + badge('v' + (s.version || '1')) + (by ? ' ' + by : '');
  const order = [['chunks', 'chunks'], ['nodes', 'nodes'], ['edges', 'edges'], ['cards', 'cards'],
    ['distilled', 'distilled'], ['adjudicate', 'adjudicate'], ['gaps', 'gaps']];
  $('#live').innerHTML = order.map(([k, lbl]) => statEl(lbl, counts[k], rates[k])).join('')
    + `<span class="stat upd">⟳ ${dt >= 1 ? '/min over ' + Math.min(60, Math.round(dt)) + 's' : 'measuring…'}</span>`;
}

async function fillSources() {
  try {
    const s = await (await fetch('stats')).json();
    const sel = $('#srcfilter');
    if (sel) sel.innerHTML = '<option value="">all sources</option>' +
      Object.keys(s.by_source || {}).map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join('');
  } catch (e) {}
}

// ── renderers ────────────────────────────────────────────────────────────────
function setRows(html, emptyMsg) {
  const r = $('#results');
  if (!html) { r.className = 'empty'; r.textContent = emptyMsg; return; }
  r.className = ''; r.innerHTML = html;
}

function renderPassages(ps, withScore) {
  setRows(ps && ps.length ? ps.map(p => `
    <div class="p">
      <div class="meta"><span class="title">${esc(p.title) || '(untitled)'}</span>
        ${p.section ? '<span>› ' + esc(p.section) + '</span>' : ''}
        ${badge(p.source_type || '?')}
        ${withScore && p.score != null ? '<span class="score">score ' + Number(p.score).toFixed(3) + '</span>' : ''}</div>
      <div class="text">${esc(p.text)}</div>
      ${p.path_or_url ? '<div class="src">' + esc(p.path_or_url) + '</div>' : ''}
    </div>`).join('') : '', 'No passages.');
}

function renderNodes(ns) {
  setRows(ns && ns.length ? ns.map(n => `
    <div class="p">
      <div class="meta"><span class="title">${esc(n.label)}</span>${badge(n.kind || 'concept')}
        ${(n.aliases && n.aliases.length) ? '<span>aka ' + esc(n.aliases.join(', ')) + '</span>' : ''}</div>
      <div class="text">${esc(n.summary)}</div>
      ${(n.sources && n.sources.length) ? '<div class="src">distilled from: ' + esc(n.sources.join(' · ')) + '</div>' : ''}
    </div>`).join('') : '', 'No distilled concepts yet — run:  python -m knowledgehost distill');
}

function renderEdges(es) {
  setRows(es && es.length ? es.map(e => {
    const meta = (e.meta || []).map(m => {
      const warn = m.type === 'disagrees_with';
      return `<div class="meta-link ${warn ? 'warn' : ''}">${warn ? '⚠ ' : '↔ '}${esc(m.type)} — ${esc(m.other)}</div>`;
    }).join('');
    return `<div class="p">
      <div class="rel"><span>${esc(e.src)}</span> <span class="arrow">—${esc(e.type)}→</span> <span>${esc(e.dst)}</span></div>
      <div class="meta">${badge(e.family)}${badge(e.regime)}${e.polarity ? badge(e.polarity) : ''}
        ${e.mechanism ? '<span>via ' + esc(e.mechanism) + '</span>' : ''}
        ${e.conditions ? '<span>if ' + esc(e.conditions) + '</span>' : ''}</div>
      ${(e.support && e.support.length) ? '<div class="src">support: ' + esc(e.support.join(' · ')) + '</div>' : ''}
      ${meta}
    </div>`;
  }).join('') : '', 'No relations yet — distil some sources.');
}

function renderCards(cs) {
  setRows(cs && cs.length ? cs.map(c => `
    <div class="p">
      <div class="meta"><span class="title">${esc(c.title)}</span>${badge(c.regime)}
        ${c.node ? '<span>node: ' + esc(c.node) + '</span>' : ''}</div>
      ${c.goal ? '<div class="text">Goal: ' + esc(c.goal) + '</div>' : ''}
      ${(c.steps && c.steps.length) ? '<ol class="steps">' + c.steps.map(s => '<li>' + esc(s) + '</li>').join('') + '</ol>' : ''}
      ${(c.support && c.support.length) ? '<div class="src">support: ' + esc(c.support.join(' · ')) + '</div>' : ''}
    </div>`).join('') : '', 'No procedure cards yet (the how-extractor lands in M2+).');
}

function renderTable(rows, cols, emptyMsg) {
  if (!rows || !rows.length) return setRows('', emptyMsg);
  const head = '<tr>' + cols.map(c => `<th>${esc(c[1])}</th>`).join('') + '</tr>';
  const body = rows.map(r => '<tr>' + cols.map(c => {
    let v = r[c[0]];
    if (typeof v === 'number') v = (c[0] === 'trust_weight' || c[0] === 'similarity') ? v : v;
    return `<td>${esc(v)}</td>`;
  }).join('') + '</tr>').join('');
  setRows('<table>' + head + body + '</table>');
}

async function load(kind) {
  $('#banner').innerHTML = ''; $('#results').className = 'empty'; $('#results').textContent = 'loading…';
  try {
    if (kind === 'raw') {
      const src = $('#srcfilter') ? $('#srcfilter').value : '';
      const res = await (await fetch(`sample?n=25${src ? '&source_type=' + encodeURIComponent(src) : ''}`)).json();
      return renderPassages(res.passages, false);
    }
    const res = await (await fetch(`browse?kind=${kind}&n=100`)).json();
    const rows = res.rows || [];
    if (kind === 'nodes') return renderNodes(rows);
    if (kind === 'edges') return renderEdges(rows);
    if (kind === 'cards') return renderCards(rows);
    if (kind === 'sources') return renderTable(rows,
      [['doc_id', 'doc'], ['title', 'title'], ['source_type', 'type'],
       ['trust_weight', 'trust'], ['regime', 'regime'], ['status', 'status']], 'No sources registered yet.');
    if (kind === 'adjudication') return renderTable(rows,
      [['node_a', 'node A'], ['node_b', 'node B'], ['similarity', 'sim'],
       ['reason', 'reason'], ['status', 'status']], 'Adjudication queue empty.');
    if (kind === 'gaps') return renderTable(rows,
      [['query_text', 'query'], ['intent', 'intent'], ['count', 'count'], ['status', 'status']],
      'No knowledge gaps logged.');
  } catch (e) { $('#results').textContent = 'request failed: ' + e; }
}

const BAND_COLOR = { high: '#22aa66', medium: '#e0a800', low: '#f5a623', contra: '#d9534f', none: '#888' };

function fchip(d, color) {
  return `<span class="badge" style="background:${color}22;border-color:${color}66">${esc(d.feature)}: ${esc(d.value)}</span>`;
}
function renderCriteria(c) {
  if (!c) return '';
  let h = '<div class="crit">';
  if (c.required && c.required.length) h += `<div class="crit-row"><b>must have</b>${c.required.map(d => fchip(d, '#22aa66')).join(' ')}</div>`;
  if (c.supportive && c.supportive.length) h += `<div class="crit-row"><b>may have</b>${c.supportive.map(d => fchip(d, '#4a90d9')).join(' ')}</div>`;
  if (c.exclusion && c.exclusion.length) h += `<div class="crit-row"><b>must NOT have</b>${c.exclusion.map(d => fchip(d, '#d9534f')).join(' ')}</div>`;
  if (c.threshold) h += `<div class="crit-row"><b>rule</b>${esc(c.threshold)}</div>`;
  if (c.gold_standard) h += `<div class="crit-row"><b>confirm</b>${esc(c.gold_standard)}</div>`;
  if (c.differentials && c.differentials.length) h += `<div class="crit-row"><b>differential</b>${c.differentials.map(d => esc(d.condition) + (d.discriminator ? ' <span style="opacity:.6">(' + esc(d.discriminator) + ')</span>' : '')).join(' · ')}</div>`;
  if (c.levels && c.levels.length) h += `<div class="crit-row"><b>stages</b>${c.levels.map(l => '<b style="opacity:1">' + esc(l.level) + '</b> ' + esc(l.label || '')).join(' · ')}</div>`;
  return h + '</div>';
}
function renderGrade(g) {
  if (!g) return '';
  return `<div class="src">recommendation: <b>${esc(g.strength || '?')}</b>${g.evidence_quality ? ' · evidence ' + esc(g.evidence_quality) : ''}${g.population ? ' · ' + esc(g.population) : ''}</div>`;
}
function renderFinding(f) {
  if (!f) return '';
  const p = ['study_design', 'direction', 'effect_size', 'n', 'certainty']
    .filter(k => f[k]).map(k => esc((k === 'n' ? 'n=' : '') + f[k]));
  return `<div class="src">finding: ${p.join(' · ')}</div>`;
}
function renderItem(it) {
  const contra = (it.contradictions || []).map(x =>
    `<div class="meta-link warn">⚠ disagrees: ${esc(x.claim)}${x.support && x.support.length ? ' [' + esc(x.support.join(', ')) + ']' : ''}</div>`).join('');
  const steps = (it.steps && it.steps.length) ? '<ol class="steps">' + it.steps.map(s => '<li>' + esc(s) + '</li>').join('') + '</ol>' : '';
  const rf = (it.red_flags && it.red_flags.length) ? `<div class="src" style="color:#d9534f">⚠ red flags: ${it.red_flags.map(esc).join(' · ')}</div>` : '';
  return `<div class="p">
    <div class="meta">${badge(it.kind)}${it.card_type ? badge(it.card_type) : ''}${badge(it.regime || 'empirical')}
      ${it.provenance ? badge(it.provenance) : ''}
      ${it.strength != null ? badge('strength ' + Number(it.strength).toFixed(3)) : ''}
      ${it.score != null ? '<span class="score">score ' + Number(it.score).toFixed(3) + '</span>' : ''}</div>
    <div class="text">${esc(it.text) || esc(it.label)}</div>
    ${renderCriteria(it.criteria)}${steps}${rf}${renderGrade(it.grade)}${renderFinding(it.finding)}
    ${(it.support && it.support.length) ? '<div class="src">support: ' + esc(it.support.join(' · ')) + '</div>' : ''}
    ${contra}
  </div>`;
}
function renderStructure(structure) {
  if (!structure || !structure.length) return '';
  return '<div class="struct"><div class="struct-h">structure — how this connects</div>' + structure.map(s =>
    `<div class="struct-sec"><span class="struct-role">${esc(s.label)}</span>`
    + s.entries.map(e => `<span class="struct-ent" title="${esc(e.text || '')}">`
      + (e.relation ? '<span style="opacity:.55">' + esc(e.relation) + '</span> ' : '')
      + esc(e.label) + (e.has_card ? ' <span style="opacity:.5">[card]</span>' : '') + '</span>').join('')
    + '</div>').join('') + '</div>';
}
function renderBundle(b) {
  const c = BAND_COLOR[b.grounding] || '#888';
  $('#banner').innerHTML =
    `<div class="note" style="background:${c}22;border-color:${c}66">`
    + (b.speech_act ? `read as <b>${esc(b.speech_act)}</b>${b.broaden ? ' <span style="opacity:.7">(ambiguous — showing the map)</span>' : ''} · ` : '')
    + `intent <b>${esc(b.intent)}</b> · rigor <b>${esc(b.rigor)}</b> · confidence ${Number(b.confidence).toFixed(3)} · `
    + `grounding <b>${esc(b.grounding)}</b>${b.abstain ? ' · <b>ABSTAIN</b>' : ''}${b.rights && b.rights.redistributable === false ? ' · <b>not redistributable</b>' : ''}<br>${esc(b.note)}</div>`;
  const items = b.items || [];
  const focus = b.focus ? `<div class="focus"><span class="focus-l">${esc(b.focus.label || '')}</span>`
    + (b.focus.summary ? '<div class="focus-s">' + esc(b.focus.summary) + '</div>' : '') + '</div>' : '';
  const html = focus + items.map(renderItem).join('') + renderStructure(b.structure);
  setRows(items.length || b.focus ? html : '', b.abstain ? 'No grounded answer — logged as a gap.' : 'No items.');
}

async function doAsk() {
  const q = $('#aq').value.trim(); if (!q) return;
  const rigor = $('#rigor') ? $('#rigor').value : '';
  $('#banner').innerHTML = ''; $('#results').className = 'empty'; $('#results').textContent = 'asking…';
  try {
    const res = await (await fetch(`ask?q=${encodeURIComponent(q)}${rigor ? '&rigor=' + rigor : ''}`)).json();
    if (!res.ok) { $('#results').textContent = 'error: ' + (res.error || 'unknown'); return; }
    renderBundle(res);
  } catch (e) { $('#results').textContent = 'request failed: ' + e; }
}

async function doSearch() {
  const q = $('#q').value.trim(); if (!q) return;
  const k = $('#k').value || 8;
  $('#banner').innerHTML = ''; $('#results').className = 'empty'; $('#results').textContent = 'searching…';
  try {
    const res = await (await fetch(`search?q=${encodeURIComponent(q)}&k=${k}`)).json();
    if (!res.ok) { $('#results').textContent = 'error: ' + (res.error || 'unknown'); return; }
    $('#banner').innerHTML =
      `<div class="note" style="background:${res.low_confidence ? '#f5a62322' : '#22aa6622'};border-color:${res.low_confidence ? '#f5a62366' : '#22aa6666'}">`
      + `confidence ${Number(res.confidence).toFixed(3)} · ${res.passages.length} passage(s) · dense ${res.dense_used ? 'on' : 'off'}`
      + (res.low_confidence ? ' · <b>low confidence</b>' : '') + '</div>';
    renderPassages(res.passages, true);
  } catch (e) { $('#results').textContent = 'request failed: ' + e; }
}

// ── control panel: Operations + Settings (auth-gated) ────────────────────────
const TOKKEY = 'kh_token';
const tok = () => localStorage.getItem(TOKKEY) || '';
const setTok = v => localStorage.setItem(TOKKEY, v || '');
function authFetch(path, opts = {}) {
  const h = Object.assign({}, opts.headers || {});
  const t = tok(); if (t) h['Authorization'] = 'Bearer ' + t;
  return fetch(path, Object.assign({}, opts, { headers: h }));
}
function postJSON(path, body) {
  return authFetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                           body: JSON.stringify(body || {}) }).then(r => r.json());
}
const tokInput = () => `<input id="tok" type="password" placeholder="auth token" value="${esc(tok())}"
  title="auth token — saved in this browser only" style="width:130px" onchange="setTok(this.value)">`;

let OPSPEC = {}, opsTimer = null, SETVALS = {};

function renderOpOptions(cmd) {
  const spec = OPSPEC[cmd] || {};
  return Object.entries(spec).map(([opt, t]) => {
    const id = 'op_' + opt;
    if (t === 'bool') return `<label class="op" style="font-size:13px"><input type="checkbox" id="${id}"> ${opt}</label>`;
    if (t === 'int') return `<label class="op" style="font-size:13px">${opt} <input type="number" id="${id}" style="width:64px"></label>`;
    if (t.startsWith('choice:')) return `<label class="op" style="font-size:13px">${opt} <select id="${id}">`
      + t.split(':')[1].split(',').map(v => `<option>${v}</option>`).join('') + `</select></label>`;
    return '';
  }).join(' ');
}
function onCmdChange() { const c = $('#opcmd'); if (c) $('#opopts').innerHTML = renderOpOptions(c.value); }
function gatherArgs() {
  const cmd = $('#opcmd').value, spec = OPSPEC[cmd] || {}, args = {};
  for (const [opt, t] of Object.entries(spec)) {
    const el = $('#op_' + opt); if (!el) continue;
    if (t === 'bool') { if (el.checked) args[opt] = true; }
    else if (t === 'int') { if (el.value !== '') args[opt] = parseInt(el.value, 10); }
    else args[opt] = el.value;
  }
  return { command: cmd, args };
}
function healthStrip(h) {
  if (!h) return '';
  const busy = h.lease_fast || h.lease_big;
  const c = busy ? '#e0a800' : '#22aa66';
  return `<span class="badge" style="background:${c}33;border-color:${c}66">GPU `
    + (busy ? 'busy — Vinkona' + (h.lease_fast ? ' · fast' : '') + (h.lease_big ? ' · big' : '') : 'free') + '</span>';
}
async function pollOps() {
  let r; try { r = await (await authFetch('/ops/log?tail=400')).json(); } catch (e) { return; }
  if (!r.ok) { $('#opstatus').textContent = (r.error === 'unauthorized')
    ? 'enter the auth token above to use Operations' : ('error: ' + r.error); return; }
  const s = r.status || {};
  const run = s.running ? `▶ ${esc(s.command)} ${esc((s.argv || []).join(' '))} — ${s.elapsed_s}s`
    : (s.command ? `■ ${esc(s.command)} finished (exit ${s.exit_code})` : 'idle — pick a command and Run');
  $('#opstatus').innerHTML = run + ' &nbsp; ' + healthStrip(r.health);
  const log = $('#opslog'); if (log) { const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.textContent = r.log || '(no output yet)'; if (atBottom) log.scrollTop = log.scrollHeight; }
}
async function loadOps() {
  $('#results').className = '';
  $('#results').innerHTML = `<div id="opstatus" style="margin:4px 0;font-size:13px">…</div>
    <pre id="opslog" style="background:#0c0c0c;color:#d8d8d8;padding:10px;border-radius:6px;
      height:46vh;overflow:auto;font-size:12px;white-space:pre-wrap;margin:0"></pre>`;
  let r; try { r = await (await authFetch('/ops/status')).json(); } catch (e) { $('#opstatus').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#opstatus').textContent = 'enter the auth token above to use Operations'; return; }
  OPSPEC = r.commands || {};
  const sel = $('#opcmd');
  if (sel) { sel.innerHTML = Object.keys(OPSPEC).sort().map(c => `<option>${c}</option>`).join(''); onCmdChange(); }
  pollOps();
}
async function runOp() {
  const body = gatherArgs();
  $('#opstatus').textContent = 'launching ' + body.command + '…';
  const r = await postJSON('/ops/run', body).catch(e => ({ ok: false, error: '' + e }));
  if (!r.ok) { $('#opstatus').textContent = '✗ ' + (r.error || 'failed'); return; }
  pollOps();
}
async function stopOp() { await postJSON('/ops/stop').catch(() => {}); pollOps(); }
async function reloadKB() {
  $('#opstatus').textContent = 'reloading KB caches…';
  const r = await postJSON('/ops/reload').catch(e => ({ ok: false, error: '' + e }));
  $('#opstatus').textContent = r.ok ? '✓ KB reloaded' : ('✗ ' + (r.error || 'failed'));
}

// ── document library: toggle which subfolders of the trusted root get indexed ──
let LIBCFG = {};
async function loadLibrary() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading library…';
  let r; try { r = await (await authFetch('/library/config')).json(); }
  catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to manage the Library'; return; }
  LIBCFG = r;
  if (!r.root) {
    $('#results').className = 'empty';
    $('#results').innerHTML = '<div style="padding:12px;line-height:1.5">No <code>library_root</code> is set. '
      + 'Add e.g. <code>library_root = "~/Library"</code> to <code>config.toml</code> on the server and restart, then Refresh.<br>'
      + '<span style="opacity:.6">The root itself is file-only, never web-editable — the panel only toggles subfolders under it.</span></div>';
    return;
  }
  if (!r.root_exists) {
    $('#results').className = 'empty';
    $('#results').innerHTML = `<div style="padding:12px">library_root <code>${esc(r.root)}</code> is not a directory on the server.</div>`;
    return;
  }
  const rows = (r.subdirs || []).map(s =>
    `<tr><td><label style="cursor:pointer"><input type="checkbox" class="libchk" value="${esc(s.name)}" ${s.active ? 'checked' : ''}> <code>${esc(s.name)}</code></label></td>`
    + `<td style="opacity:.55">${s.active ? 'indexed' : ''}</td></tr>`).join('')
    || '<tr><td colspan=2 style="opacity:.5">no subfolders under the root</td></tr>';
  const collOpts = ['<option value="">all collections</option>']
    .concat((r.subdirs || []).map(s => `<option value="${esc(s.name)}">${esc(s.name)}</option>`)).join('');
  $('#results').innerHTML =
    `<div style="margin:6px 0 10px;font-size:13px">Trusted root <code>${esc(r.root)}</code> `
    + `<span style="opacity:.6">— tick the subfolders (each becomes a search <b>collection</b>), Save, then run <b>ingest-library</b> in Operations.</span></div>`
    + `<table><tr><th>subfolder / collection</th><th></th></tr>${rows}</table>`
    + `<hr style="margin:16px 0;border:none;border-top:1px solid #8884">`
    + `<div style="font-size:13px;margin-bottom:6px"><b>Test search</b> — the exact ranked path Vinkona's research loop calls (BM25${r && r.dense ? ' + dense' : ''} → rerank).</div>`
    + `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">`
    + `<input id="libq" type="search" placeholder="query the indexed library…" style="flex:1;min-width:220px" onkeydown="if(event.key==='Enter')doLibrarySearch()">`
    + `<select id="libcoll" title="restrict to a collection">${collOpts}</select>`
    + `<input id="libk" type="number" value="8" min="1" max="50" style="width:64px" title="results">`
    + `<button class="toolbtn" onclick="doLibrarySearch()">Search</button></div>`
    + `<div id="libbanner"></div><div id="libresults" style="opacity:.6;font-size:13px">Index a selection, then search it here to eyeball ranking + latency.</div>`;
}
async function doLibrarySearch() {
  const q = ($('#libq').value || '').trim(); if (!q) return;
  const coll = $('#libcoll').value || '', k = $('#libk').value || 8;
  $('#libbanner').innerHTML = ''; $('#libresults').style.opacity = 1; $('#libresults').textContent = 'searching…';
  let url = `library?q=${encodeURIComponent(q)}&k=${k}`;
  if (coll) url += `&collection=${encodeURIComponent(coll)}`;
  const t0 = performance.now();
  let res; try { res = await (await fetch(url)).json(); }
  catch (e) { $('#libresults').textContent = 'request failed: ' + e; return; }
  const ms = Math.round(performance.now() - t0);
  if (!res.ok) { $('#libresults').textContent = 'error: ' + (res.error || 'unknown'); return; }
  const ps = res.passages || [];
  $('#libbanner').innerHTML =
    `<div class="note" style="background:${res.low_confidence ? '#f5a62322' : '#22aa6622'};border-color:${res.low_confidence ? '#f5a62366' : '#22aa6666'}">`
    + `confidence ${Number(res.confidence).toFixed(3)} · ${ps.length} passage(s) · dense ${res.dense_used ? 'on' : 'off'} · ${ms} ms`
    + (res.low_confidence ? ' · <b>no match</b>' : '') + '</div>';
  $('#libresults').innerHTML = ps.length ? ps.map(p => `
    <div class="p">
      <div class="meta"><span class="title">${esc(p.title) || '(untitled)'}</span>
        ${p.section ? '<span>› ' + esc(p.section) + '</span>' : ''}
        ${badge(p.collection || '?')}
        ${p.score != null ? '<span class="score">score ' + Number(p.score).toFixed(3) + '</span>' : ''}</div>
      <div class="text">${esc(p.text)}</div>
      ${p.path_or_url ? '<div class="src">' + esc(p.path_or_url) + '</div>' : ''}
    </div>`).join('') : '<div class="empty" style="padding:10px">No match in the local library.</div>';
}
async function saveLibrarySelection() {
  const active = Array.from(document.querySelectorAll('.libchk:checked')).map(c => c.value);
  $('#banner').innerHTML = 'saving…';
  const r = await postJSON('/library/config', { active }).catch(e => ({ ok: false, error: '' + e }));
  if (!r.ok) { $('#banner').innerHTML = `<span style="color:#c00">✗ ${esc(r.error || 'failed')}</span>`; return; }
  $('#banner').innerHTML = '<span style="color:#0a0">✓ saved — now run <b>ingest-library</b> in Operations to (re)index</span>';
  loadLibrary();
}

// ── modular knowledge: Bundles (groups, scenarios, source rename/regroup) ────
let BUNDLES = {};
async function loadBundles() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading bundles…';
  let r; try { r = await (await authFetch('/bundles')).json(); } catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to manage Bundles'; return; }
  BUNDLES = r;
  const scen = $('#scensel');
  if (scen) {
    const names = Array.from(new Set(['all', ...Object.keys(r.scenarios || {})]));
    scen.innerHTML = names.map(n => `<option ${n === r.active ? 'selected' : ''}>${esc(n)}</option>`).join('');
  }
  const enc = new Set(r.encrypted_bundles || []);
  const grp = (r.bundles || []).map(b =>
    `<tr><td><code>${esc(b.bundle)}</code>${enc.has(b.bundle) ? ' 🔒' : ''}</td><td>${b.sources}</td></tr>`).join('')
    || '<tr><td colspan=2 style="opacity:.5">no sources yet</td></tr>';
  const scenRows = Object.entries(r.scenarios || {}).map(([n, d]) =>
    `<tr><td><code>${esc(n)}</code>${n === r.active ? ' <span class="badge">active</span>' : ''}</td>`
    + `<td style="opacity:.7">${d.include ? 'include ' + esc((d.include || []).join(', ')) : ''}`
    + `${d.exclude ? ' exclude ' + esc((d.exclude || []).join(', ')) : ''}</td></tr>`).join('')
    || '<tr><td colspan=2 style="opacity:.5">none defined — edit config.toml [scenarios.NAME]</td></tr>';
  const bopts = (r.bundles || []).map(b => esc(b.bundle));
  const SPDX = ['CC0-1.0','public-domain','CC-BY-4.0','CC-BY-SA-4.0','CC-BY-NC-4.0',
    'CC-BY-NC-SA-4.0','CC-BY-ND-4.0','CC-BY-NC-ND-4.0','MIT','Apache-2.0','GPL-3.0-only',
    'proprietary','all-rights-reserved','unknown'];
  const srcRows = (r.sources || []).map(s => {
    const b = s.bundle || 'base';
    return `<tr><td style="max-width:190px;overflow:hidden;text-overflow:ellipsis" title="${esc(s.doc_id)}"><code>${esc(s.doc_id)}</code></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="title" value="${esc(s.title || '')}" style="width:150px"></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="bundle" value="${esc(b)}" list="blist" style="width:96px"></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="license" value="${esc(s.license || '')}" list="spdxlist" placeholder="unknown" style="width:130px"></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="license_holder" value="${esc(s.license_holder || '')}" placeholder="licensor" style="width:150px"></td>`
      + `<td><button class="toolbtn" onclick="saveSource('${esc(s.doc_id)}')">save</button></td></tr>`;
  }).join('') || '<tr><td colspan=6 style="opacity:.5">no sources</td></tr>';
  $('#results').innerHTML =
    `<datalist id="blist">${bopts.map(b => `<option value="${b}">`).join('')}</datalist>`
    + `<datalist id="spdxlist">${SPDX.map(b => `<option value="${b}">`).join('')}</datalist>`
    + `<div style="opacity:.6;margin-bottom:8px">serving <b>${esc(r.active)}</b>`
    + (r.modular ? ` — working DB <code>${esc((r.working || '').split('/').pop())}</code>` : ' — master (no scenario)')
    + `</div>`
    + `<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">`
    + `<div><h4 style="margin:4px 0">Bundles</h4><table><tr><th>bundle</th><th>sources</th></tr>${grp}</table></div>`
    + `<div><h4 style="margin:4px 0">Scenarios</h4><table><tr><th>name</th><th>rule</th></tr>${scenRows}</table></div>`
    + `</div>`
    + `<h4 style="margin:14px 0 4px">Sources — rename, assign a bundle, set the licence &amp; licensor (writes to master)</h4>`
    + `<table><tr><th>doc_id</th><th>title</th><th>bundle</th><th>licence</th><th>licensor (to whom)</th><th></th></tr>${srcRows}</table>`;
}
async function saveSource(doc) {
  const g = f => document.querySelector(`[data-doc="${CSS.escape(doc)}"][data-f="${f}"]`).value;
  const r = await postJSON('/source', { doc_id: doc, title: g('title'), bundle: g('bundle'),
    license: g('license'), license_holder: g('license_holder') }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok ? '✓ saved ' + esc(doc) + ' — ' + esc(r.note || '') : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) loadBundles();
}
async function applyScenario() {
  const name = $('#scensel') ? $('#scensel').value : '';
  $('#banner').innerHTML = `<div class="note">reassembling scenario '${esc(name)}' — this may take a moment…</div>`;
  const r = await postJSON('/scenario', { scenario: name }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? "✓ now serving '" + esc(r.scenario) + "' — " + (r.counts ? JSON.stringify(r.counts) : '')
    : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) { loadBundles(); refreshStats(); }
}

// ── Prioritizer: the autopilot plan (ordered auto-run of maintenance verbs) ──
let APLAN = null, APSPEC = {};
async function loadAutopilot() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading plan…';
  let r; try { r = await (await authFetch('/ops/autopilot')).json(); } catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to use the Prioritizer'; return; }
  APLAN = r.plan; APSPEC = r.commands || {};
  renderAutopilot(r.state || {});
}
function renderAutopilot(state) {
  const p = APLAN;
  const st = state.enabled
    ? `<b style="color:#2e7d32">ON</b> — ${esc(state.running_step || state.last_reason || 'idle')}`
    : `<b style="color:#999">off</b>`;
  const rows = (p.steps || []).map((s, i) => {
    const opts = Object.keys(APSPEC).sort().map(c =>
      `<option ${c === s.command ? 'selected' : ''}>${c}</option>`).join('');
    return `<tr data-i="${i}">
      <td style="white-space:nowrap">
        <button class="toolbtn" onclick="moveStep(${i},-1)" title="higher priority" ${i === 0 ? 'disabled' : ''}>▲</button>
        <button class="toolbtn" onclick="moveStep(${i},1)" title="lower priority" ${i === p.steps.length - 1 ? 'disabled' : ''}>▼</button>
      </td>
      <td><input type="checkbox" data-f="enabled" ${s.enabled ? 'checked' : ''}></td>
      <td><select data-f="command">${opts}</select></td>
      <td><input data-f="args" value="${esc(JSON.stringify(s.args || {}))}" style="width:180px"
                 title='JSON, e.g. {"bundle":"vinkona"} or {"limit":50}'></td>
      <td><input data-f="min_interval_s" type="number" min="0" value="${s.min_interval_s || 0}" style="width:90px"></td>
      <td><input data-f="label" value="${esc(s.label || '')}" style="width:220px"></td>
      <td><button class="toolbtn" onclick="delStep(${i})">✕</button></td></tr>`;
  }).join('');
  $('#results').innerHTML =
    `<div style="margin:6px 0 12px;font-size:13px">
       <label><input type="checkbox" id="apEnabled" ${p.enabled ? 'checked' : ''}> <b>Autopilot enabled</b></label>
       &nbsp;·&nbsp; status: ${st}
       <div style="opacity:.65;margin-top:6px">Steps run top-to-bottom by priority; after each, the list is
         re-checked from the top, so a higher step that just gained work (e.g. fresh Vinkona drops) preempts
         the backlog below it. “Min interval” throttles a step; leave 0 to run whenever there's work.</div>
     </div>
     <label style="font-size:13px"><input type="checkbox" id="apLeases" ${p.respect_leases ? 'checked' : ''}>
       Yield to the assistant (pause while it's using the LMs)</label>
     &nbsp;·&nbsp; <label style="font-size:13px">idle re-check
       <input id="apInterval" type="number" min="5" value="${p.idle_interval_s || 60}" style="width:70px">s</label>
     <table style="margin-top:10px"><tr><th>order</th><th>on</th><th>command</th><th>args (JSON)</th>
       <th>min interval s</th><th>label</th><th></th></tr>${rows}</table>`;
}
function _readAutopilotForm() {
  const steps = [];
  document.querySelectorAll('#results tr[data-i]').forEach(tr => {
    const g = f => tr.querySelector(`[data-f="${f}"]`);
    let args = {};
    try { args = JSON.parse(g('args').value || '{}'); } catch (e) { args = {}; }
    steps.push({ command: g('command').value, enabled: g('enabled').checked,
                 args, min_interval_s: parseInt(g('min_interval_s').value || '0', 10),
                 label: g('label').value });
  });
  return { enabled: $('#apEnabled').checked, respect_leases: $('#apLeases').checked,
           idle_interval_s: parseInt($('#apInterval').value || '60', 10), steps };
}
function moveStep(i, d) { const s = APLAN.steps; const j = i + d;
  if (j < 0 || j >= s.length) return; APLAN = _readAutopilotForm();
  [APLAN.steps[i], APLAN.steps[j]] = [APLAN.steps[j], APLAN.steps[i]]; renderAutopilot({}); }
function delStep(i) { APLAN = _readAutopilotForm(); APLAN.steps.splice(i, 1); renderAutopilot({}); }
function addAutopilotStep() { APLAN = _readAutopilotForm();
  APLAN.steps.push({ command: 'distill', args: {}, enabled: true, min_interval_s: 0, label: 'new step' });
  renderAutopilot({}); }
async function saveAutopilot() {
  const plan = _readAutopilotForm();
  const r = await postJSON('/ops/autopilot', { plan }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok ? '✓ plan saved' : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) loadAutopilot();
}

async function loadSettings() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading config…';
  let r; try { r = await (await authFetch('/config')).json(); } catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to edit Settings'; return; }
  const sch = r.schema || {}; SETVALS = r.values || {};
  const rows = Object.keys(sch).sort().map(k => {
    const t = sch[k].type, v = SETVALS[k];
    const inp = t === 'bool'
      ? `<input type="checkbox" data-k="${k}" ${v ? 'checked' : ''}>`
      : `<input data-k="${k}" value="${esc(v)}" style="width:170px">`;
    return `<tr><td><code>${esc(k)}</code></td><td style="opacity:.6">${t}</td><td>${inp}</td>`
      + `<td style="opacity:.45">${esc(sch[k].default)}</td></tr>`;
  }).join('');
  $('#results').innerHTML = `<div style="opacity:.6;margin-bottom:6px">writing to ${esc(r.config_path || '(server started without -c — read-only)')}</div>`
    + `<table><tr><th>key</th><th>type</th><th>value</th><th>default</th></tr>${rows}</table>`;
}
async function saveSettings() {
  const updates = {};
  document.querySelectorAll('#results [data-k]').forEach(el => {
    const k = el.dataset.k;
    const v = el.type === 'checkbox' ? el.checked : el.value;
    const changed = el.type === 'checkbox' ? (v !== !!SETVALS[k]) : (String(v) !== String(SETVALS[k] == null ? '' : SETVALS[k]));
    if (changed) updates[k] = v;
  });
  if (!Object.keys(updates).length) { $('#banner').innerHTML = '<div class="note">no changes</div>'; return; }
  const r = await postJSON('/config', { updates }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? '✓ saved ' + Object.keys(r.applied || {}).length + ' setting(s) — ' + esc(r.note || '')
    : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) loadSettings();
}

buildTabs(); refreshStats(); setInterval(refreshStats, REFRESH_MS); loadHelp(); go('ask');
</script>
</body>
</html>
"""
