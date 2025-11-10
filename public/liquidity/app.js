document.addEventListener('DOMContentLoaded', () => {
  if (typeof attachGlobalErrorHandlers === 'function') {
    attachGlobalErrorHandlers();
  } else if (!window.__ncErrorHandlerAttached) {
    window.__ncErrorHandlerAttached = true;
    window.addEventListener('error', (event) => {
      if (!event) { return; }
      console.error('[NeoCortex UI]', event.message, event.error);
    });
    window.addEventListener('unhandledrejection', (event) => {
      if (!event) { return; }
      console.error('[NeoCortex UI]', event.reason);
    });
  }

  const form = document.getElementById('queryForm');
  const statusEl = document.getElementById('status');
  const submitBtn = document.getElementById('submitBtn');
  const sweepList = document.getElementById('sweepList');
  const clusterList = document.getElementById('clusterList');
  const heatmapContainer = document.getElementById('heatmapContainer');
  const summary = {
    date: document.getElementById('summaryDate'),
    bars: document.getElementById('summaryBars'),
    volume: document.getElementById('summaryVolume'),
    range: document.getElementById('summaryRange'),
    bull: document.getElementById('summaryBull'),
    bear: document.getElementById('summaryBear'),
    net: document.getElementById('summaryNet')
  };

  if (!form || !statusEl || !submitBtn || !sweepList || !clusterList || !heatmapContainer) {
    console.error('[NeoCortex UI] Liquidity radar elements missing');
    return;
  }

  const dateInput = document.getElementById('date');
  if (dateInput) {
    const today = new Date();
    const iso = today.toISOString().slice(0, 10);
    dateInput.value = iso;
    dateInput.max = iso;
  }

  function fmtNumber(value, { digits = 2, compact = false, sign = false } = {}) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  const opts = { maximumFractionDigits: digits, minimumFractionDigits: digits };
  if (compact) opts.notation = 'compact';
  if (sign) opts.signDisplay = 'always';
  return Number(value).toLocaleString(undefined, opts);
}

function clearList(listEl, message) {
  listEl.innerHTML = '';
  const li = document.createElement('li');
  li.className = 'empty';
  li.textContent = message;
  listEl.appendChild(li);
}

function renderSweeps(sweeps) {
  sweepList.innerHTML = '';
  if (!sweeps || sweeps.length === 0) {
    clearList(sweepList, 'No sweeps detected for this session.');
    return;
  }
  sweeps.slice(0, 40).forEach((sweep) => {
    const li = document.createElement('li');
    const badge = document.createElement('span');
    badge.className = `badge ${sweep.direction === 'bullish' ? 'bull' : 'bear'}`;
    badge.textContent = sweep.direction === 'bullish' ? 'Bull sweep' : 'Bear sweep';
    const ts = document.createElement('strong');
    ts.textContent = new Date(sweep.time).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    const detail = document.createElement('span');
    detail.className = 'muted';
    detail.textContent = `Volume× ${fmtNumber(sweep.volume_ratio, { digits: 1 })} · Range× ${fmtNumber(sweep.range_ratio, { digits: 1 })}`;
    const wick = document.createElement('span');
    wick.className = 'muted';
    wick.textContent = `Wick ${fmtNumber(sweep.wick, { digits: 3 })} | Body ${fmtNumber(sweep.body, { digits: 3 })}`;
    li.appendChild(badge);
    li.appendChild(ts);
    li.appendChild(detail);
    li.appendChild(wick);
    sweepList.appendChild(li);
  });
}

function renderClusters(clusters) {
  clusterList.innerHTML = '';
  if (!clusters || clusters.length === 0) {
    clearList(clusterList, 'No manipulation clusters within ±20 min detected.');
    return;
  }
  clusters.forEach((cluster) => {
    const li = document.createElement('li');
    const title = document.createElement('strong');
    const from = new Date(cluster.start).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    const to = new Date(cluster.end).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    title.textContent = `${from} → ${to}`;
    const count = document.createElement('span');
    count.className = 'muted';
    count.textContent = `${cluster.count} sweeps (${cluster.directions.join(' → ')})`;
    li.appendChild(title);
    li.appendChild(count);
    clusterList.appendChild(li);
  });
}

function renderSummary(analysis) {
  const session = analysis.session || {};
  const footprint = (analysis.footprint && analysis.footprint.totals) || {};
  summary.date.textContent = session.date || '—';
  summary.bars.textContent = fmtNumber(session.bars, { digits: 0 });
  summary.volume.textContent = fmtNumber(analysis.orderflow?.volume, { compact: true, digits: 2 });
  summary.range.textContent = fmtNumber(analysis.orderflow?.average_range, { digits: 3 });
  summary.bull.textContent = fmtNumber(Math.max(footprint.max_delta || 0, 0), { compact: true, digits: 2 });
  summary.bear.textContent = fmtNumber(Math.min(footprint.min_delta || 0, 0), { compact: true, digits: 2, sign: true });
  summary.net.textContent = fmtNumber(footprint.delta, { compact: true, digits: 2, sign: true });
}

  function renderHeatmap(info) {
  if (!info || !info.public_url) {
    heatmapContainer.className = 'empty';
    heatmapContainer.textContent = 'No heatmap generated for this session.';
    return;
  }
  const img = document.createElement('img');
  img.src = `${info.public_url}?t=${Date.now()}`;
  img.alt = 'Liquidity heatmap';
  heatmapContainer.className = '';
  heatmapContainer.innerHTML = '';
  heatmapContainer.appendChild(img);
}

  async function runQuery(event) {
  event.preventDefault();
  const formData = new FormData(form);
  const ticker = formData.get('ticker');
  const date = formData.get('date');
  const interval = formData.get('interval');
  if (!ticker) {
    return;
  }
  submitBtn.disabled = true;
  statusEl.textContent = 'Scanning session…';
  statusEl.className = 'empty';

  const params = new URLSearchParams({ ticker, interval });
  if (date) params.set('session_date', date);

  try {
    const resp = await fetch(`/strategy/liquidity-sweeps?${params.toString()}`, {
      headers: { 'X-Requested-With': 'fetch' }
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.detail || 'Scan failed');
    }
    statusEl.textContent = `Analysed ${data.session?.bars ?? 0} bars from ${data.session?.start ?? ''}`;
    statusEl.className = 'muted';
    renderSummary(data);
    renderSweeps(data.sweeps);
    renderClusters(data.manipulation_clusters);
    renderHeatmap(data.heatmap);
  } catch (err) {
    console.error(err);
    statusEl.textContent = err.message || 'Unable to complete scan';
    statusEl.className = 'empty';
    clearList(sweepList, 'No data');
    clearList(clusterList, 'No data');
    heatmapContainer.className = 'empty';
    heatmapContainer.textContent = 'Heatmap unavailable.';
  } finally {
    submitBtn.disabled = false;
  }
  }

  form.addEventListener('submit', runQuery);
});
