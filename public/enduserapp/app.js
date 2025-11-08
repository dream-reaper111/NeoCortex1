const webhookUrlEl = document.getElementById('webhookUrl');
const copyBtn = document.getElementById('copyWebhook');
const form = document.getElementById('testForm');
const responseOutput = document.getElementById('responseOutput');
const statusEl = document.getElementById('testStatus');
const refreshBtn = document.getElementById('refreshTests');
const artifactList = document.getElementById('artifactList');
const yearEl = document.getElementById('year');

function computeWebhookUrl() {
  const { origin } = window.location;
  const url = `${origin.replace(/\/$/, '')}/alpaca/webhook`;
  webhookUrlEl.textContent = url;
  webhookUrlEl.dataset.url = url;
}

function copyWebhook() {
  const url = webhookUrlEl.dataset.url;
  if (!url) return;
  navigator.clipboard.writeText(url)
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

async function sendTest(event) {
  event.preventDefault();
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

  // remove null optional fields to keep payload clean
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

function boot() {
  computeWebhookUrl();
  yearEl.textContent = new Date().getFullYear();
  loadArtifacts();
}

copyBtn.addEventListener('click', copyWebhook);
form.addEventListener('submit', sendTest);
refreshBtn.addEventListener('click', loadArtifacts);
window.addEventListener('load', boot);
