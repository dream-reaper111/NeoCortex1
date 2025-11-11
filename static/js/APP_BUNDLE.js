document.addEventListener('DOMContentLoaded', () => {
    // Mobile navigation toggle
    document.querySelectorAll('[data-nav]').forEach((nav) => {
        const toggle = nav.querySelector('[data-nav-toggle]');
        const panel = nav.querySelector('[data-nav-panel]');
        if (!toggle || !panel) return;
        toggle.addEventListener('click', () => {
            const expanded = toggle.getAttribute('aria-expanded') === 'true';
            toggle.setAttribute('aria-expanded', String(!expanded));
            panel.classList.toggle('is-open', !expanded);
        });

        // Highlight active link
        const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
        nav.querySelectorAll('.nc-nav__link').forEach((link) => {
            const href = (link.getAttribute('href') || '').replace(/\/+$/, '') || '/';
            if (href === currentPath) {
                link.classList.add('is-active');
            }
        });
    });

    // Dashboard tab controls
    document.querySelectorAll('[data-tabs]').forEach((tabs) => {
        const buttons = Array.from(tabs.querySelectorAll('[data-tab]'));
        const panels = buttons
            .map((btn) => document.querySelector(btn.dataset.tab))
            .filter(Boolean);
        function activate(target) {
            buttons.forEach((btn) => {
                const active = btn.dataset.tab === target;
                btn.classList.toggle('is-active', active);
                btn.setAttribute('aria-selected', String(active));
            });
            panels.forEach((panel) => {
                panel.classList.toggle('is-active', '#' + panel.id === target);
            });
        }
        buttons.forEach((btn) => {
            btn.addEventListener('click', () => activate(btn.dataset.tab));
        });
        if (buttons.length) {
            activate(buttons[0].dataset.tab);
        }
    });

    const logoutBtn = document.getElementById('globalLogoutBtn');
    if (logoutBtn) {
        const TOKEN_KEY = 'nc_token';
        const USERNAME_KEY = 'nc_user';
        const COOKIE_NAME_KEY = 'nc_cookie';
        const REFRESH_KEY = 'nc_refresh';
        logoutBtn.addEventListener('click', async (event) => {
            event.preventDefault();
            logoutBtn.disabled = true;
            try {
                const headers = {};
                if (window.CSRF) {
                    try {
                        const token = await window.CSRF.ensureToken();
                        if (token) {
                            headers[window.CSRF.headerName] = token;
                        }
                    } catch (err) {
                        console.warn('Unable to refresh CSRF token before logout', err);
                    }
                }
                await fetch('/logout', { method: 'POST', credentials: 'include', headers });
            } catch (_) {
                /* swallow */
            }
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USERNAME_KEY);
            localStorage.removeItem(REFRESH_KEY);
            const cookieName = localStorage.getItem(COOKIE_NAME_KEY) || 'session_token';
            document.cookie = cookieName + '=; Max-Age=0; path=/';
            window.location.href = '/login';
        });
    }

    // Initialize basic charts if Chart.js is available
    if (window.Chart) {
        document.querySelectorAll('[data-chart="radar"] canvas').forEach((canvas) => {
            const ctx = canvas.getContext('2d');
            new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Momentum', 'Liquidity', 'Sweep', 'Dark Pool', 'Options', 'Volume'],
                    datasets: [{
                        label: 'NeoCortex Pulse',
                        data: [60, 72, 55, 68, 62, 74],
                        backgroundColor: 'rgba(0, 212, 255, 0.15)',
                        borderColor: '#00d4ff',
                        pointBackgroundColor: '#b84dff',
                        borderWidth: 2
                    }]
                },
                options: {
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            ticks: { display: false },
                            grid: { color: 'rgba(0, 212, 255, 0.15)' },
                            angleLines: { color: 'rgba(184, 77, 255, 0.24)' }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        });
    }
});
