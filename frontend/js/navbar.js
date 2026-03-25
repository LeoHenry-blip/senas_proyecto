/* ============================================================
   js/navbar.js — Navbar compartido + toggle de tema
   Incluir en TODOS los HTML con:
   <script src="js/navbar.js"></script>

   MODO SOLO TEMA (sub-páginas en iframe):
   Agrega data-theme-only al <body> para que no renderice
   el navbar, solo aplique y sincronice el tema.
   <body data-theme-only>
   ============================================================ */
(function () {
  const THEME_KEY = 'senas-v2-theme';

  // ---- Variables de cada tema ----
  const DARK = {
    '--bg-900': '#070b12', '--bg-800': '#0d1117',
    '--bg-700': '#161c27', '--bg-600': '#1e2736',
    '--bg-500': '#273348', '--border': '#2a3550',
    '--border-light': '#3a4f70', '--text-100': '#eaf0fa',
    '--text-200': '#b0bcce', '--text-300': '#6b7fa0',
    '--shadow': '0 8px 40px rgba(0,0,0,.6)',
    '--shadow-sm': '0 2px 12px rgba(0,0,0,.4)',
    '--grad-hero': 'linear-gradient(135deg,rgba(99,102,241,.12),rgba(20,184,166,.07))',
  };
  const LIGHT = {
    '--bg-900': '#f4f6fb', '--bg-800': '#ffffff',
    '--bg-700': '#f8fafd', '--bg-600': '#eef2f8',
    '--bg-500': '#e4eaf4', '--border': '#dde3ef',
    '--border-light': '#c5cfe0', '--text-100': '#111827',
    '--text-200': '#374151', '--text-300': '#6b7280',
    '--shadow': '0 8px 40px rgba(0,0,0,.12)',
    '--shadow-sm': '0 2px 12px rgba(0,0,0,.08)',
    '--grad-hero': 'linear-gradient(135deg,rgba(99,102,241,.08),rgba(20,184,166,.05))',
  };

  // ---- Aplica el tema inyectando variables en <html> ----
  function applyTheme(isLight) {
    const vars = isLight ? LIGHT : DARK;
    const root = document.documentElement;
    Object.entries(vars).forEach(([k, v]) => root.style.setProperty(k, v));
    document.body.classList.toggle('light-mode', isLight);
    document.body.dataset.theme = isLight ? 'light' : 'dark';
    // Actualiza el ícono del botón (puede no existir aún al llamarse)
    const btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.textContent = isLight ? '☀️' : '🌙';
      btn.title = isLight ? 'Cambiar a modo oscuro' : 'Cambiar a modo claro';
    }
  }

  // ---- Inyecta el navbar en el <body> ----
  function renderNavbar() {
    // Modo solo-tema: sub-página en iframe, no necesita navbar
    if (document.body.hasAttribute('data-theme-only')) return;

    // Si la página ya tiene su propio navbar estático, NO lo duplica
    if (document.querySelector('.navbar')) return;

    // Sin navbar propio → crea uno genérico
    const nav = document.createElement('nav');
    nav.className = 'navbar';
    nav.innerHTML = `
      <span class="navbar-brand">🤟 Señas V2</span>
      <div class="navbar-nav">
        <span id="nav-nombre" class="text-muted text-sm"></span>
        <span id="nav-badge-admin"></span>
        <button id="theme-toggle" class="btn btn-ghost btn-sm theme-btn" title="Modo claro">🌙</button>
        <button class="btn btn-ghost btn-sm" onclick="Auth && Auth.logout()">Salir</button>
      </div>
    `;
    document.body.insertBefore(nav, document.body.firstChild);
  }

  // ---- Engancha el botón de tema con delegación de eventos ----
  // Delegación en document: funciona sin importar cuándo
  // se crea el botón ni si el DOM aún no está completo.
  function setupToggle() {
    document.addEventListener('click', (e) => {
      if (e.target.closest('#theme-toggle')) {
        const willBeLight = document.body.dataset.theme !== 'light';
        localStorage.setItem(THEME_KEY, willBeLight ? 'light' : 'dark');
        applyTheme(willBeLight);
      }
    });
  }

  // ---- Sincroniza entre pestañas e iframes (StorageEvent) ----
  window.addEventListener('storage', (e) => {
    if (e.key === THEME_KEY) {
      applyTheme(e.newValue === 'light');
    }
  });

  // ---- Init ----
  function init() {
    renderNavbar();
    setupToggle();
    // Aplica el tema guardado (dark por defecto)
    applyTheme(localStorage.getItem(THEME_KEY) === 'light');
  }

  // Ejecuta cuando el DOM esté listo
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();