const webhookUrlEl = document.getElementById('webhookUrl');
const copyBtn = document.getElementById('copyWebhook');
const form = document.getElementById('testForm');
const responseOutput = document.getElementById('responseOutput');
const statusEl = document.getElementById('testStatus');
const refreshBtn = document.getElementById('refreshTests');
const artifactList = document.getElementById('artifactList');
const yearEl = document.getElementById('year');

const paperForm = document.getElementById('paperForm');
const paperStatus = document.getElementById('paperStatus');
const aiStatusEl = document.getElementById('aiStatus');
const pnlTotalEl = document.getElementById('pnlTotal');
const pnlLongEl = document.getElementById('pnlLong');
const pnlShortEl = document.getElementById('pnlShort');
const pnlTimestampEl = document.getElementById('pnlTimestamp');
const longTable = document.getElementById('longTable');
const shortTable = document.getElementById('shortTable');
const instrumentSelect = document.getElementById('paperInstrument');
const optionFields = Array.from(document.querySelectorAll('.option-field'));
const futureFields = Array.from(document.querySelectorAll('.future-field'));

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const LIVE_PNL_REFRESH_MS = 15000;
let paperRefreshTimer = null;

function computeWebhookUrl() {
  if (!webhookUrlEl) return;
  const { origin } = window.location;
  const url = `${origin.replace(/\/$/, '')}/alpaca/webhook`;
  webhookUrlEl.textContent = url;
  webhookUrlEl.dataset.url = url;
}

function copyWebhook() {
  if (!webhookUrlEl || !copyBtn) return;
  const url = webhookUrlEl.dataset.url;
  if (!url) return;
  navigator.clipboard
    .writeText(url)
    .then(() => {
      copyBtn.textContent = 'Copied!';
      copyBtn.classList.add('primary');
      setTimeout(() => {
        copyBtn.textContent = 'Copy';
        copyBtn.classList.remove('primary');
      }, 1600);
    })
    .catch(() => {
      copyBtn.textContent = 'Copy failed';
      copyBtn.classList.add('primary');
    });
}

function formatCurrency(value, withSign = false) {
  const amount = Number.isFinite(value) ? Number(value) : 0;
  const formatted = currencyFormatter.format(amount);
  if (withSign && amount > 0 && !formatted.startsWith('+')) {
    return `+${formatted}`;
  }
  return formatted;
}

function setPnlClass(el, value) {
  if (!el) return;
  el.classList.remove('pnl-positive', 'pnl-negative', 'pnl-neutral');
  if (value > 0.0001) {
    el.classList.add('pnl-positive');
  } else if (value < -0.0001) {
    el.classList.add('pnl-negative');
  } else {
    el.classList.add('pnl-neutral');
  }
}

function instrumentDescription(order) {
  if (!order) return '—';
  if (order.instrument === 'option') {
    const parts = [order.option_type ? order.option_type.toUpperCase() : null];
    if (order.strike != null) {
      parts.push(`@ ${Number(order.strike).toFixed(2)}`);
    }
    if (order.expiry) {
      parts.push(order.expiry);
    }
    return `Option ${parts.filter(Boolean).join(' ') || ''}`.trim();
  }
  if (order.instrument === 'future') {
    const month = order.future_month ? order.future_month.toUpperCase() : '—';
    const year = order.future_year || '';
    return `Future ${month} ${year}`.trim();
  }
  return String(order.instrument || '—').toUpperCase();
}

function renderPositions(tableEl, orders) {
  if (!tableEl) return;
  const tbody = tableEl.querySelector('tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!orders || !orders.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.className = 'empty';
    cell.textContent = 'No positions yet.';
    row.append(cell);
    tbody.append(row);
    return;
  }

  for (const order of orders) {
    const row = document.createElement('tr');

    const symbolCell = document.createElement('td');
    const symbol = document.createElement('strong');
    symbol.textContent = order.symbol || '—';
    symbolCell.append(symbol);
    if (order.quantity != null) {
      const qty = document.createElement('span');
      qty.className = 'table-subtext';
      qty.textContent = `Qty ${Number(order.quantity).toFixed(2)}`;
      symbolCell.append(qty);
    }

    const instrumentCell = document.createElement('td');
    instrumentCell.textContent = instrumentDescription(order);

    const entryCell = document.createElement('td');
    entryCell.textContent = formatCurrency(order.entry_price ?? order.price ?? 0);

    const lastCell = document.createElement('td');
    lastCell.textContent = formatCurrency(order.current_price ?? order.price ?? 0);

    const pnlValue = Number(order.pnl ?? 0);
    const pnlCell = document.createElement('td');
    pnlCell.className = 'pnl-cell';
    pnlCell.textContent = formatCurrency(pnlValue, true);
    setPnlClass(pnlCell, pnlValue);

    const statusCell = document.createElement('td');
    const pill = document.createElement('span');
    const status = order.status === 'executed' ? 'executed' : 'review';
    pill.className = `status-pill ${status}`;
    pill.textContent = status === 'executed' ? 'AI Executed' : 'AI Review';
    statusCell.append(pill);
    if (order.ai && typeof order.ai.confidence !== 'undefined') {
      const detail = document.createElement('span');
      detail.className = 'table-subtext';
      detail.textContent = `Confidence ${Number(order.ai.confidence).toFixed(2)}`;
      statusCell.append(detail);
    }

    row.append(symbolCell, instrumentCell, entryCell, lastCell, pnlCell, statusCell);
    tbody.append(row);
  }
}

function updatePnlWidget(pnl, timestamp) {
  if (!pnlTotalEl || !pnlLongEl || !pnlShortEl || !pnlTimestampEl) return;
  const totals = pnl || {};
  const total = Number(totals.total || 0);
  const longValue = Number(totals.long || 0);
  const shortValue = Number(totals.short || 0);

  pnlTotalEl.textContent = formatCurrency(total);
  pnlLongEl.textContent = formatCurrency(longValue, true);
  pnlShortEl.textContent = formatCurrency(shortValue, true);

  setPnlClass(pnlTotalEl, total);
  setPnlClass(pnlLongEl, longValue);
  setPnlClass(pnlShortEl, shortValue);

  if (timestamp) {
    try {
      pnlTimestampEl.textContent = new Date(timestamp).toLocaleTimeString();
    } catch (err) {
      pnlTimestampEl.textContent = '--';
    }
  } else {
    pnlTimestampEl.textContent = '--';
  }
}

function syncInstrumentFields() {
  if (!instrumentSelect) return;
  const value = instrumentSelect.value;
  const isOption = value === 'option';
  optionFields.forEach((el) => {
    el.classList.toggle('hidden', !isOption);
  });
  futureFields.forEach((el) => {
    el.classList.toggle('hidden', isOption);
  });
}

async function sendTest(event) {
  event.preventDefault();
  if (!form) return;
  statusEl.textContent = 'Sending…';
  statusEl.classList.remove('error');

  const data = {
    symbol: form.symbol.value.trim() || 'SPY',
    quantity: Number(form.quantity.value) || 0,
    price: Number(form.price.value) || 0,
    side: form.side.value,
    status: form.status.value,
    event: form.event.value,
    timestamp: form.timestamp.value.trim() || null,
  };

  const rawText = form.raw.value.trim();
  if (rawText) {
    try {
      data.raw = JSON.parse(rawText);
    } catch (err) {
      statusEl.textContent = 'Invalid JSON in Raw payload';
      statusEl.classList.add('error');
      return;
    }
  }

  if (!data.timestamp) {
    delete data.timestamp;
  }
  if (!data.raw) {
    delete data.raw;
  }

  try {
    const res = await fetch('/alpaca/webhook/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const body = await res.json();
    if (!res.ok || !body.ok) {
      throw new Error(body.detail || res.statusText);
    }
    statusEl.textContent = 'Test sent successfully';
    responseOutput.textContent = JSON.stringify(body, null, 2);
    await loadArtifacts();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.classList.add('error');
    responseOutput.textContent = err.stack || err.message;
  }
}

async function loadArtifacts() {
  if (!artifactList) return;
  artifactList.innerHTML = '<li class="artifact-item"><span>Loading…</span></li>';
  try {
    const res = await fetch('/alpaca/webhook/tests');
    const body = await res.json();
    if (!res.ok || !body.ok) {
      throw new Error(body.detail || 'Unable to load artifacts');
    }
    const items = body.tests;
    if (!items.length) {
      artifactList.innerHTML = '<li class="artifact-item"><span>No tests recorded yet.</span></li>';
      return;
    }
    artifactList.innerHTML = '';
    for (const item of items) {
      const li = document.createElement('li');
      li.className = 'artifact-item';
      const info = document.createElement('div');
      info.innerHTML = `<strong>${item.symbol || item.name}</strong> <span class="artifact-item__meta">${item.created}</span>`;
      const link = document.createElement('a');
      link.href = item.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = 'Download JSON';
      li.append(info, link);
      artifactList.append(li);
    }
  } catch (err) {
    artifactList.innerHTML = `<li class="artifact-item"><span class="artifact-item__meta">${err.message}</span></li>`;
  }
}

async function loadPaperDashboard() {
  if (!paperForm) return;
  try {
    const res = await fetch('/papertrade/status');
    const body = await res.json();
    if (!res.ok || !body.ok) {
      throw new Error(body.detail || 'Unable to load paper trading status');
    }
    const dashboard = body.dashboard || {};
    updatePnlWidget(dashboard.pnl || {}, dashboard.generated_at);
    if (dashboard.orders) {
      renderPositions(longTable, dashboard.orders.long || []);
      renderPositions(shortTable, dashboard.orders.short || []);
    }
    return dashboard;
  } catch (err) {
    console.warn('[papertrade] refresh failed', err);
    if (pnlTimestampEl) {
      pnlTimestampEl.textContent = 'offline';
    }
    if (aiStatusEl && !aiStatusEl.textContent) {
      aiStatusEl.textContent = 'Neo Cortex AI is awaiting connection…';
    }
    return null;
  }
}

function sanitizeOrderPayload(payload) {
  const data = { ...payload };
  for (const key of Object.keys(data)) {
    const value = data[key];
    if (value === '' || value === null || Number.isNaN(value)) {
      delete data[key];
    }
  }
  return data;
}

async function submitPaperOrder(event) {
  event.preventDefault();
  if (!paperForm) return;
  if (paperRefreshTimer) {
    clearInterval(paperRefreshTimer);
    paperRefreshTimer = null;
  }

  if (paperStatus) {
    paperStatus.textContent = 'Submitting…';
    paperStatus.classList.remove('error');
  }

  const sideInput = paperForm.querySelector('input[name="paperSide"]:checked');
  const payload = {
    instrument: instrumentSelect ? instrumentSelect.value : 'option',
    symbol: paperForm.symbol.value.trim(),
    quantity: Number(paperForm.quantity.value) || 0,
    price: Number(paperForm.price.value) || 0,
    side: sideInput ? sideInput.value : 'long',
  };

  if (payload.instrument === 'option') {
    payload.option_type = paperForm.option_type.value;
    payload.expiry = paperForm.expiry.value;
    payload.strike = paperForm.strike.value ? Number(paperForm.strike.value) : null;
  } else {
    payload.future_month = paperForm.future_month.value.trim();
    payload.future_year = paperForm.future_year.value ? Number(paperForm.future_year.value) : null;
  }

  try {
    if (!payload.symbol) {
      throw new Error('Symbol is required');
    }
    if (payload.quantity <= 0) {
      throw new Error('Quantity must be greater than zero');
    }

    const bodyPayload = sanitizeOrderPayload(payload);
    const res = await fetch('/papertrade/orders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bodyPayload),
    });
    const body = await res.json();
    if (!res.ok || !body.ok) {
      throw new Error(body.detail || res.statusText || 'Order rejected');
    }

    if (paperStatus) {
      paperStatus.textContent = 'Order sent to Neo Cortex AI.';
    }
    if (aiStatusEl && body.order && body.order.ai) {
      const ai = body.order.ai;
      aiStatusEl.textContent = `${ai.message} (confidence ${Number(ai.confidence).toFixed(2)})`;
    }
    const dashboard = body.dashboard || {};
    updatePnlWidget(dashboard.pnl || {}, dashboard.generated_at);
    if (dashboard.orders) {
      renderPositions(longTable, dashboard.orders.long || []);
      renderPositions(shortTable, dashboard.orders.short || []);
    }
  } catch (err) {
    if (paperStatus) {
      paperStatus.textContent = `Error: ${err.message}`;
      paperStatus.classList.add('error');
    }
    if (aiStatusEl) {
      aiStatusEl.textContent = 'Neo Cortex AI could not process the order.';
    }
  } finally {
    paperRefreshTimer = setInterval(loadPaperDashboard, LIVE_PNL_REFRESH_MS);
  }
}

function boot() {
  computeWebhookUrl();
  if (yearEl) {
    yearEl.textContent = new Date().getFullYear();
  }
  loadArtifacts();
  if (paperForm) {
    syncInstrumentFields();
    loadPaperDashboard();
    paperRefreshTimer = setInterval(loadPaperDashboard, LIVE_PNL_REFRESH_MS);
  }
}

if (copyBtn) {
  copyBtn.addEventListener('click', copyWebhook);
}
if (form) {
  form.addEventListener('submit', sendTest);
}
if (refreshBtn) {
  refreshBtn.addEventListener('click', loadArtifacts);
}
if (paperForm) {
  paperForm.addEventListener('submit', submitPaperOrder);
}
if (instrumentSelect) {
  instrumentSelect.addEventListener('change', syncInstrumentFields);
}

window.addEventListener('load', boot);
