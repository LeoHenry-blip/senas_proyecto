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

 function renderNavbar() {
  if (document.body.hasAttribute('data-theme-only')) return;
  if (document.querySelector('.navbar')) return;

  const LOGO_URL = 'https://cdn-icons-png.flaticon.com/512/8750/8750693.png';
  const LEARN_URL = 'https://www.strongasl.com/?lang=es';

  const nav = document.createElement('nav');
  nav.className = 'navbar';

  nav.innerHTML = `
    <span class="navbar-brand" style="display:flex;align-items:center;gap:.65rem;">
      <span style="display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,rgba(99,102,241,.18),rgba(20,184,166,.12));border:1px solid rgba(99,102,241,.25);box-shadow:0 0 12px rgba(99,102,241,.2);flex-shrink:0;padding:5px;">
        <img src="${LOGO_URL}" alt="Logo" style="width:100%;height:100%;object-fit:contain;filter:drop-shadow(0 0 4px rgba(99,102,241,.5));" onerror="this.parentElement.style.display='none';">
      </span>
      Sistema Inteligente de Traducción de Señas
    </span>
    <div class="navbar-nav">
      <span id="nav-nombre" class="text-muted text-sm"></span>
      <span id="nav-badge-admin"></span>
      <a id="btn-aprender" href="${LEARN_URL}" target="_blank" rel="noopener noreferrer">
        🔆 Diccionario de Señas ¡Aprende señas!
      </a>
      <button id="theme-toggle" class="btn btn-ghost btn-sm theme-btn" title="Modo claro">🌙</button>
      <button class="btn btn-ghost btn-sm" onclick="Auth && Auth.logout()">Salir</button>
    </div>
  `;

  /* Estilos del botón aplicados por JS — evita el problema de comillas en innerHTML */
  document.body.insertBefore(nav, document.body.firstChild);

  const btnAprender = nav.querySelector('#btn-aprender');
  Object.assign(btnAprender.style, {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '.4rem',
    fontSize: '.82rem',
    fontWeight: '600',
    color: '#a5b4fc',
    textDecoration: 'none',
    padding: '.35rem .85rem',
    borderRadius: '8px',
    border: '1px solid rgba(99,102,241,.25)',
    background: 'rgba(99,102,241,.1)',
    transition: 'background .2s, border-color .2s, transform .15s',
    whiteSpace: 'nowrap',
    cursor: 'pointer',
  });

  btnAprender.addEventListener('mouseover', () => {
    btnAprender.style.background    = 'rgba(99,102,241,.22)';
    btnAprender.style.borderColor   = 'rgba(99,102,241,.5)';
    btnAprender.style.transform     = 'translateY(-1px)';
  });
  btnAprender.addEventListener('mouseout', () => {
    btnAprender.style.background    = 'rgba(99,102,241,.1)';
    btnAprender.style.borderColor   = 'rgba(99,102,241,.25)';
    btnAprender.style.transform     = 'translateY(0)';
  });
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