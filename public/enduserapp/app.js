const TOKEN_KEY = 'nc_token';
const USERNAME_KEY = 'nc_user';
const COOKIE_NAME_KEY = 'nc_cookie';

const authOverlay = document.getElementById('authOverlay');
const authForm = document.getElementById('authForm');
const authMessage = document.getElementById('authMessage');
const authRegisterBtn = document.getElementById('authRegisterBtn');
const authWhopBtn = document.getElementById('authWhopBtn');
const authWhopNotice = document.getElementById('authWhopNotice');
const authWhopExtras = document.getElementById('authWhopExtras');
const authAccountType = document.getElementById('authAccountType');
const authApiKey = document.getElementById('authApiKey');
const authApiSecret = document.getElementById('authApiSecret');
const authBaseUrl = document.getElementById('authBaseUrl');
const authUsername = document.getElementById('authUsername');
const authPassword = document.getElementById('authPassword');
const userBadge = document.getElementById('userBadge');
const logoutBtn = document.getElementById('logoutBtn');

const params = new URLSearchParams(window.location.search);

if (!localStorage.getItem(COOKIE_NAME_KEY)) {
  localStorage.setItem(COOKIE_NAME_KEY, 'session_token');
}

const LIVE_PNL_REFRESH_MS = 15000;
let paperRefreshTimer = null;
let unauthorizedHandled = false;
let whopMode = false;
let whopSessionValid = false;

function resolveNext(candidate) {
  if (!candidate) {
    return window.location.pathname + window.location.search + window.location.hash;
  }
  try {
    const parsed = new URL(candidate, window.location.origin);
    if (parsed.origin !== window.location.origin) {
      return window.location.pathname + window.location.search + window.location.hash;
    }
    return parsed.pathname + parsed.search + parsed.hash;
  } catch (_err) {
    return window.location.pathname + window.location.search + window.location.hash;
  }
}

const nextUrl = resolveNext(params.get('next'));
const whopToken = (params.get('whop_token') || '').trim();
const DEFAULT_SIGNIN_MESSAGE = 'Sign in with your Whop membership to continue.';

function authToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function sessionCookieName() {
  return localStorage.getItem(COOKIE_NAME_KEY) || 'session_token';
}

function clearSession() {
  const cookieName = sessionCookieName();
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USERNAME_KEY);
  if (cookieName) {
    document.cookie = `${cookieName}=; Max-Age=0; path=/`;
  }
}

function setAuthMessage(text, type) {
  if (!authMessage) return;
  authMessage.textContent = text || '';
  authMessage.className = 'auth-message' + (type ? ` ${type}` : '');
}

function showWhopNotice(text, type) {
  if (!authWhopNotice) return;
  if (!text) {
    authWhopNotice.textContent = '';
    authWhopNotice.className = 'auth-notice';
    authWhopNotice.hidden = true;
    return;
  }
  authWhopNotice.textContent = text;
  authWhopNotice.className = 'auth-notice' + (type ? ` ${type}` : '');
  authWhopNotice.hidden = false;
}

function showOverlay(message, type) {
  if (authOverlay) {
    authOverlay.hidden = false;
  }
  document.body.classList.add('auth-locked');
  if (typeof message === 'string') {
    setAuthMessage(message, type);
  } else {
    setAuthMessage(DEFAULT_SIGNIN_MESSAGE, '');
  }
  if (whopMode && authForm && !authForm.hidden && authUsername) {
    authUsername.focus();
  } else if (authWhopBtn && !authWhopBtn.hidden) {
    authWhopBtn.focus();
  }
}

function hideOverlay() {
  if (authOverlay) {
    authOverlay.hidden = true;
  }
  document.body.classList.remove('auth-locked');
  setAuthMessage('', '');
}

function updateUserBadge() {
  const username = localStorage.getItem(USERNAME_KEY) || 'member';
  const authed = Boolean(authToken());
  if (userBadge) {
    userBadge.textContent = authed ? `Signed in as ${username}` : 'Whop sign-in required';
  }
  if (logoutBtn) {
    logoutBtn.hidden = !authed;
  }
}

function handleUnauthorized() {
  if (unauthorizedHandled) return;
  unauthorizedHandled = true;
  clearSession();
  updateUserBadge();
  showOverlay('Your session has expired. Sign in again to continue.', 'error');
  stopApp();
}

async function sendAuth(path, body) {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (_err) {
    throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
  }
  if (!resp.ok || !data.ok) {
    throw new Error(data.detail || data.reason || 'Request failed');
  }
  return data;
}

function completeAuthentication(data, fallbackUsername) {
  unauthorizedHandled = false;
  const username = data.username || fallbackUsername || 'admin';
  localStorage.setItem(TOKEN_KEY, data.token);
  localStorage.setItem(USERNAME_KEY, username);
  localStorage.setItem(COOKIE_NAME_KEY, data.session_cookie || 'session_token');
  updateUserBadge();
  hideOverlay();
  startApp();
}

(function patchFetch() {
  const originalFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const token = authToken();
    const headers = new Headers(init.headers || {});
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', 'Bearer ' + token);
    }
    const finalInit = Object.assign({}, init, { headers });
    if (!finalInit.credentials) {
      finalInit.credentials = 'include';
    }
    return originalFetch(input, finalInit).then((response) => {
      if (response.status === 401) {
        handleUnauthorized();
      }
      return response;
    });
  };
})();

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
const positionsStatus = document.getElementById('positionsStatus');
const instrumentSelect = document.getElementById('paperInstrument');
const optionFields = Array.from(document.querySelectorAll('.option-field'));
const futureFields = Array.from(document.querySelectorAll('.future-field'));

function stopApp() {
  if (paperRefreshTimer) {
    clearInterval(paperRefreshTimer);
    paperRefreshTimer = null;
  }
}

function startApp() {
  stopApp();
  boot();
}

if (authRegisterBtn) {
  authRegisterBtn.hidden = true;
  authRegisterBtn.dataset.mode = '';
}

if (authForm) {
  authForm.hidden = true;
}

if (authWhopExtras) {
  authWhopExtras.hidden = true;
}

if (authWhopBtn) {
  authWhopBtn.hidden = true;
  authWhopBtn.disabled = true;
  authWhopBtn.dataset.bound = '';
}

showWhopNotice('', '');

if (whopToken) {
  whopMode = true;
  if (authRegisterBtn) {
    authRegisterBtn.hidden = true;
    authRegisterBtn.dataset.mode = 'whop';
    authRegisterBtn.textContent = 'Complete Membership Setup';
    authRegisterBtn.disabled = true;
  }
  if (authWhopBtn) {
    authWhopBtn.hidden = true;
  }
  if (authForm) {
    authForm.hidden = true;
  }
  showWhopNotice('Verifying your Whop membership…');
  (async () => {
    try {
      const resp = await fetch(`/auth/whop/session?token=${encodeURIComponent(whopToken)}`, {
        credentials: 'include',
      });
      let data = null;
      try {
        data = await resp.json();
      } catch (_err) {
        data = null;
      }
      if (!resp.ok || !data || data.ok === false) {
        whopSessionValid = false;
        const detail =
          (data && (data.detail || data.reason)) ||
          'Unable to verify your Whop membership. Please restart from Whop.';
        showWhopNotice(detail, 'error');
        if (authRegisterBtn) {
          authRegisterBtn.hidden = true;
          authRegisterBtn.disabled = true;
        }
        if (authForm) {
          authForm.hidden = true;
        }
        setAuthMessage(detail, 'error');
        return;
      }
      const email = data.email || '';
      const existingUsername = data.username || '';
      if (data.registered) {
        const welcome = existingUsername
          ? `Welcome back, ${existingUsername}!`
          : email
          ? `Welcome back, ${email}!`
          : 'Membership verified!';
        showWhopNotice(`${welcome} Signing you in…`, 'success');
        setAuthMessage('Signing you in…');
        try {
          const loginData = await sendAuth('/auth/whop/login', { token: whopToken });
          setAuthMessage('Welcome back! Loading your console…', 'success');
          completeAuthentication(loginData, existingUsername || (email ? email.split('@')[0] : 'member'));
        } catch (err) {
          console.error(err);
          showWhopNotice('Unable to sign you in automatically. Please start again from Whop.', 'error');
          setAuthMessage(err.message || 'Whop sign-in failed', 'error');
        }
        return;
      }
      if (authForm) {
        authForm.hidden = false;
      }
      if (authWhopExtras) {
        authWhopExtras.hidden = false;
      }
      if (authRegisterBtn) {
        authRegisterBtn.hidden = false;
        authRegisterBtn.disabled = false;
      }
      const greeting = email ? `Welcome, ${email}!` : 'Whop membership verified!';
      showWhopNotice(`${greeting} Complete your funding credentials to finish setup.`, 'success');
      if (email && authUsername && !authUsername.value) {
        authUsername.value = email.split('@')[0];
      }
      setAuthMessage('Choose your console credentials and link your Alpaca account to continue.', '');
      whopSessionValid = true;
    } catch (err) {
      console.error(err);
      whopSessionValid = false;
      showWhopNotice('Unable to verify your Whop membership. Please restart from Whop.', 'error');
      if (authRegisterBtn) {
        authRegisterBtn.hidden = true;
        authRegisterBtn.disabled = true;
      }
      if (authForm) {
        authForm.hidden = true;
      }
      setAuthMessage('Unable to verify your Whop membership. Please restart from Whop.', 'error');
    }
  })();
} else {
  whopMode = false;
  whopSessionValid = false;
  if (authForm) {
    authForm.hidden = true;
  }
  if (authWhopExtras) {
    authWhopExtras.hidden = true;
  }
  if (authRegisterBtn) {
    authRegisterBtn.hidden = true;
    authRegisterBtn.dataset.mode = '';
    authRegisterBtn.disabled = false;
  }
  setAuthMessage(DEFAULT_SIGNIN_MESSAGE, '');
}

async function setupWhopButton() {
  if (!authWhopBtn || whopMode) return;
  try {
    const resp = await fetch('/auth/whop/start?mode=status', { credentials: 'include' });
    let data = null;
    try {
      data = await resp.json();
    } catch (_err) {
      data = null;
    }
    if (!resp.ok || !data || data.enabled !== true) {
      authWhopBtn.hidden = true;
      authWhopBtn.disabled = true;
      setAuthMessage('Whop sign-in is not currently available. Contact support for assistance.', 'error');
      return;
    }
  } catch (err) {
    console.warn('Unable to determine Whop availability', err);
    authWhopBtn.hidden = true;
    authWhopBtn.disabled = true;
    setAuthMessage('Unable to reach Whop. Check your connection or try again later.', 'error');
    return;
  }

  if (!authWhopBtn.dataset.bound) {
    authWhopBtn.addEventListener('click', () => {
      const dest = '/auth/whop/start?next=' + encodeURIComponent(nextUrl);
      window.location.href = dest;
    });
    authWhopBtn.dataset.bound = '1';
  }
  authWhopBtn.hidden = false;
  authWhopBtn.disabled = false;
}

setupWhopButton();

if (authForm) {
  authForm.addEventListener('submit', (event) => {
    event.preventDefault();
    if (authRegisterBtn && !authRegisterBtn.hidden && authRegisterBtn.dataset.mode === 'whop') {
      authRegisterBtn.click();
    }
  });
}

if (authRegisterBtn) {
  authRegisterBtn.addEventListener('click', async () => {
    if (!whopMode || authRegisterBtn.dataset.mode !== 'whop') {
      return;
    }
    const username = authUsername ? authUsername.value.trim().toLowerCase() : '';
    const password = authPassword ? authPassword.value : '';
    if (!username || !password) {
      setAuthMessage('Choose a username and password before completing registration.', 'error');
      return;
    }
    if (!whopSessionValid) {
      setAuthMessage('Your Whop session has expired. Start again from Whop.', 'error');
      return;
    }
    const apiKey = authApiKey ? authApiKey.value.trim() : '';
    const apiSecret = authApiSecret ? authApiSecret.value.trim() : '';
    const baseUrl = authBaseUrl ? authBaseUrl.value.trim() : '';
    if (!apiKey || !apiSecret) {
      setAuthMessage('Enter your Alpaca API key and secret to continue.', 'error');
      return;
    }
    authRegisterBtn.disabled = true;
    setAuthMessage('Linking your Whop membership…');
    try {
      const payload = {
        token: whopToken,
        username,
        password,
        api_key: apiKey,
        api_secret: apiSecret,
        account_type: (authAccountType && authAccountType.value ? authAccountType.value : 'funded').toLowerCase(),
      };
      if (baseUrl) {
        payload.base_url = baseUrl;
      }
      const data = await sendAuth('/register/whop', payload);
      setAuthMessage('Welcome aboard! Loading your console…', 'success');
      completeAuthentication(data, username);
    } catch (err) {
      setAuthMessage(err.message || 'Whop registration failed', 'error');
    } finally {
      authRegisterBtn.disabled = !whopSessionValid;
    }
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener('click', async (event) => {
    event.preventDefault();
    try {
      await fetch('/logout', { method: 'POST' });
    } catch (_err) {
      // Ignore logout network errors
    }
    clearSession();
    stopApp();
    updateUserBadge();
    showOverlay('Signed out. Sign in with Whop to continue.', '');
  });
}

updateUserBadge();
if (authToken()) {
  hideOverlay();
  startApp();
} else {
  showOverlay();
}

function getAccountType() {
  if (!paperForm) return 'paper';
  const value = (paperForm.dataset.account || 'paper').trim().toLowerCase();
  return value || 'paper';
}

function formatAccountLabel(accountType) {
  const value = (accountType || 'paper').toString();
  if (!value) return 'Paper';
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function describeAssetClass(assetClass) {
  if (!assetClass) return 'Asset';
  const normalized = String(assetClass).replace(/_/g, ' ').trim();
  if (!normalized) return 'Asset';
  return normalized.replace(/\b\w/g, (char) => char.toUpperCase());
}

function normalizeAlpacaPosition(position) {
  if (!position) return null;
  const rawQty = Number(position.qty ?? position.quantity ?? 0);
  const qty = Number.isFinite(rawQty) ? rawQty : 0;
  const sideRaw = (position.side || (qty < 0 ? 'short' : 'long') || 'long').toString().toLowerCase();
  const side = sideRaw === 'short' ? 'short' : 'long';
  const quantity = Math.abs(qty);
  const entryPriceRaw = Number(
    position.avg_entry_price ?? position.avg_entry ?? position.entry_price ?? position.price ?? 0,
  );
  const entryPrice = Number.isFinite(entryPriceRaw) ? entryPriceRaw : 0;
  const currentPriceRaw = Number(
    position.current_price ??
      position.market_price ??
      position.lastday_price ??
      position.asset_current_price ??
      position.price ??
      0,
  );
  const currentPrice = Number.isFinite(currentPriceRaw) ? currentPriceRaw : 0;
  const pnlRaw = Number(position.unrealized_pl ?? position.unrealized_intraday_pl ?? 0);
  const pnl = Number.isFinite(pnlRaw) ? pnlRaw : 0;

  return {
    symbol: (position.symbol || '').toUpperCase(),
    quantity,
    instrument: describeAssetClass(position.asset_class),
    entry_price: entryPrice,
    current_price: currentPrice,
    pnl,
    cost_basis: Number(position.cost_basis ?? 0) || 0,
    market_value: Number(position.market_value ?? 0) || 0,
    side,
    status: 'executed',
  };
}

function computePnlTotals(positions) {
  const totals = { total: 0, long: 0, short: 0 };
  if (!Array.isArray(positions)) {
    return totals;
  }
  for (const pos of positions) {
    if (!pos) continue;
    const value = Number(pos.pnl ?? 0);
    if (!Number.isFinite(value)) continue;
    const side = (pos.side || 'long').toLowerCase() === 'short' ? 'short' : 'long';
    totals.total += value;
    if (side === 'short') {
      totals.short += value;
    } else {
      totals.long += value;
    }
  }
  return totals;
}

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

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
    cell.colSpan = 7;
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

    const actionsCell = document.createElement('td');
    actionsCell.className = 'actions-cell';
    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'btn inline';
    closeBtn.textContent = 'Close';
    const orderSymbol = (order.symbol || '').trim();
    if (!orderSymbol) {
      closeBtn.disabled = true;
    } else {
      closeBtn.addEventListener('click', () => closePosition(orderSymbol, closeBtn));
    }
    actionsCell.append(closeBtn);

    row.append(symbolCell, instrumentCell, entryCell, lastCell, pnlCell, statusCell, actionsCell);
    tbody.append(row);
  }
}

function updatePositionsStatus(message, type) {
  if (!positionsStatus) return;
  positionsStatus.textContent = message || '';
  positionsStatus.classList.remove('error', 'success');
  if (type) {
    positionsStatus.classList.add(type);
  }
}

async function closePosition(symbol, button) {
  const normalizedSymbol = (symbol || '').trim();
  if (!normalizedSymbol) {
    updatePositionsStatus('Unable to close: missing symbol.', 'error');
    return;
  }

  const btn = button || null;
  if (btn) {
    btn.disabled = true;
    btn.dataset.originalText = btn.textContent;
    btn.textContent = 'Closing…';
  }

  const accountType = getAccountType();
  const accountLabel = formatAccountLabel(accountType);
  updatePositionsStatus(
    `Closing ${normalizedSymbol.toUpperCase()} on the ${accountLabel} account…`,
  );

  try {
    const res = await fetch(
      `/positions/${encodeURIComponent(normalizedSymbol)}/close?account=${encodeURIComponent(accountType)}`,
      {
        method: 'POST',
      },
    );
    let body = null;
    try {
      body = await res.json();
    } catch (_err) {
      body = null;
    }
    if (!res.ok || (body && body.ok === false)) {
      const detail = (body && body.detail) || res.statusText || 'Unable to close position';
      throw new Error(detail);
    }
    const detail =
      (body && body.detail) ||
      `Closed ${normalizedSymbol.toUpperCase()} position on the ${accountLabel} account.`;
    updatePositionsStatus(detail, 'success');
    await loadPaperDashboard();
  } catch (err) {
    console.error('[positions] close failed', err);
    updatePositionsStatus(
      `Close failed on the ${accountLabel} account: ${err.message}`,
      'error',
    );
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = btn.dataset.originalText || 'Close';
      delete btn.dataset.originalText;
    }
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

async function loadLiveDashboard(accountType) {
  if (!paperForm) return null;
  const query = accountType ? `?account=${encodeURIComponent(accountType)}` : '';
  let positionsBody = null;
  try {
    const [positionsRes, pnlRes] = await Promise.all([
      fetch(`/positions${query}`, { cache: 'no-store' }),
      fetch(`/pnl${query}`, { cache: 'no-store' }),
    ]);

    try {
      positionsBody = await positionsRes.json();
    } catch (_err) {
      positionsBody = null;
    }

    if (!positionsRes.ok || !positionsBody || positionsBody.ok === false) {
      const detail =
        (positionsBody && (positionsBody.detail || positionsBody.error)) ||
        positionsRes.statusText ||
        'Unable to load Alpaca positions';
      throw new Error(detail);
    }

    const rawPositions = Array.isArray(positionsBody.positions)
      ? positionsBody.positions
      : [];
    const normalizedPositions = rawPositions
      .map((position) => normalizeAlpacaPosition(position))
      .filter(Boolean);
    const longPositions = normalizedPositions.filter((pos) => pos.side !== 'short');
    const shortPositions = normalizedPositions.filter((pos) => pos.side === 'short');
    renderPositions(longTable, longPositions);
    renderPositions(shortTable, shortPositions);

    let totals = computePnlTotals(normalizedPositions);
    let timestamp = new Date().toISOString();

    let pnlBody = null;
    try {
      pnlBody = await pnlRes.json();
    } catch (_err) {
      pnlBody = null;
    }
    if (pnlRes.ok && pnlBody && pnlBody.ok !== false) {
      if (typeof pnlBody.total_pnl !== 'undefined') {
        const totalValue = Number(pnlBody.total_pnl);
        if (Number.isFinite(totalValue)) {
          totals.total = totalValue;
        }
      }
      if (Array.isArray(pnlBody.positions) && pnlBody.positions.length) {
        const breakdownPositions = pnlBody.positions.map((position) => {
          const qty = Number(position.quantity ?? position.qty ?? 0);
          const pnlValue = Number(position.unrealized_pl ?? 0);
          return {
            pnl: Number.isFinite(pnlValue) ? pnlValue : 0,
            side: qty < 0 ? 'short' : 'long',
          };
        });
        const breakdown = computePnlTotals(breakdownPositions);
        if (breakdown.long || breakdown.short) {
          totals.long = breakdown.long;
          totals.short = breakdown.short;
          if (!totals.total) {
            totals.total = breakdown.total;
          }
        }
      }
    } else if (!pnlRes.ok) {
      console.warn('[pnl] refresh failed', pnlBody || pnlRes.statusText || pnlRes.status);
    }

    const label = formatAccountLabel(accountType);
    updatePnlWidget(totals, timestamp);
    if (normalizedPositions.length === 0) {
      updatePositionsStatus(`No live Alpaca positions on the ${label} account.`, 'success');
    } else {
      updatePositionsStatus(
        `Displaying live Alpaca positions for the ${label} account.`,
        'success',
      );
    }

    return { positions: normalizedPositions, pnl: totals };
  } catch (err) {
    console.warn('[positions] live refresh failed', err);
    if (err && err.message) {
      updatePositionsStatus(err.message, 'error');
    }
    return null;
  }
}

async function loadMockDashboard(message, statusClass) {
  if (!paperForm) return null;
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
    if (typeof message === 'undefined') {
      updatePositionsStatus('Showing simulated paper trading positions.', 'success');
    } else if (message !== null) {
      updatePositionsStatus(message, statusClass || 'success');
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
    updatePositionsStatus('Unable to load positions.', 'error');
    return null;
  }
}

async function loadPaperDashboard() {
  if (!paperForm) return null;
  const accountType = getAccountType();
  const live = await loadLiveDashboard(accountType);
  if (live) {
    return live;
  }
  return loadMockDashboard(
    'Live Alpaca positions unavailable. Showing simulated paper trading positions.',
    'error',
  );
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
    if (paperRefreshTimer) {
      clearInterval(paperRefreshTimer);
    }
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
