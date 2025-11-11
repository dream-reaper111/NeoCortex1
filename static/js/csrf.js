(function () {
    const STORAGE_KEY = 'csrf_token';
    const HEADER_NAME = 'x-csrf-token';
    const FIELD_NAME = 'csrf_token';
    let inflight = null;

    function applyToForms(token) {
        if (!token) return;
        const forms = document.querySelectorAll('form');
        forms.forEach((form) => {
            let input = form.querySelector(`input[name="${FIELD_NAME}"][type="hidden"]`);
            if (!input) {
                input = document.createElement('input');
                input.type = 'hidden';
                input.name = FIELD_NAME;
                input.setAttribute('data-csrf-auto', 'true');
                form.appendChild(input);
            }
            input.value = token;
        });
    }

    function storeToken(token) {
        if (typeof token !== 'string' || !token) return;
        localStorage.setItem(STORAGE_KEY, token);
        applyToForms(token);
    }

    async function fetchToken(force) {
        if (inflight && !force) {
            try {
                return await inflight;
            } catch (_) {
                // Ignore and continue below
            }
        }
        inflight = (async () => {
            const resp = await fetch('/csrf-token', {
                credentials: 'include',
                headers: { Accept: 'application/json' },
                cache: 'no-store',
            });
            if (!resp.ok) {
                throw new Error(`Failed to fetch CSRF token: ${resp.status}`);
            }
            const data = await resp.json();
            if (!data || typeof data.csrf_token !== 'string') {
                throw new Error('Malformed CSRF token response');
            }
            storeToken(data.csrf_token);
            return data.csrf_token;
        })();
        try {
            return await inflight;
        } finally {
            inflight = null;
        }
    }

    async function initCsrf() {
        const existing = localStorage.getItem(STORAGE_KEY);
        if (existing) {
            applyToForms(existing);
        }
        try {
            await fetchToken(true);
        } catch (err) {
            console.warn('Unable to initialize CSRF token', err);
        }
    }

    window.CSRF = {
        init: initCsrf,
        refresh: () => fetchToken(true),
        ensureToken: () => {
            const current = localStorage.getItem(STORAGE_KEY);
            if (current) {
                return Promise.resolve(current);
            }
            return fetchToken(true);
        },
        token: () => localStorage.getItem(STORAGE_KEY),
        headerName: HEADER_NAME,
        apply: applyToForms,
    };

    document.addEventListener('DOMContentLoaded', initCsrf);
})();
