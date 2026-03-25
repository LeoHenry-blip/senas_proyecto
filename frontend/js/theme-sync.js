/**
 * theme-sync.js
 * Las sub-páginas de admin_pages/ incluyen este script.
 * Recibe el tema desde el iframe padre (admin.html) y aplica
 * las mismas variables CSS para que el modo oscuro/claro sea consistente.
 */
(function () {
  const DARK = {
    '--bg-900':'#070b12','--bg-800':'#0d1117','--bg-700':'#161c27',
    '--bg-600':'#1e2736','--bg-500':'#273348','--border':'#2a3550',
    '--border-light':'#3a4f70','--text-100':'#eaf0fa','--text-200':'#b0bcce',
    '--text-300':'#6b7fa0','--shadow':'0 8px 40px rgba(0,0,0,.6)',
    '--shadow-sm':'0 2px 12px rgba(0,0,0,.4)',
    '--grad-hero':'linear-gradient(135deg,rgba(99,102,241,.12),rgba(20,184,166,.07))',
  };
  const LIGHT = {
    '--bg-900':'#f4f6fb','--bg-800':'#ffffff','--bg-700':'#f8fafd',
    '--bg-600':'#eef2f8','--bg-500':'#e4eaf4','--border':'#dde3ef',
    '--border-light':'#c5cfe0','--text-100':'#111827','--text-200':'#374151',
    '--text-300':'#6b7280','--shadow':'0 8px 40px rgba(0,0,0,.12)',
    '--shadow-sm':'0 2px 12px rgba(0,0,0,.08)',
    '--grad-hero':'linear-gradient(135deg,rgba(99,102,241,.08),rgba(20,184,166,.05))',
  };

  function applyTheme(isLight) {
    const vars = isLight ? LIGHT : DARK;
    Object.entries(vars).forEach(([k, v]) =>
      document.documentElement.style.setProperty(k, v)
    );
    document.body.classList.toggle('light-mode', isLight);
  }

  // Aplicar el tema guardado en localStorage al cargar
  applyTheme(localStorage.getItem('senas-v2-theme') === 'light');

  // Escuchar mensaje del padre (admin.html) cuando el usuario cambia el tema
  window.addEventListener('message', (e) => {
    if (e.data?.type === 'THEME') applyTheme(e.data.isLight);
  });

  // También escuchar cambios directos en localStorage (otras pestañas)
  window.addEventListener('storage', (e) => {
    if (e.key === 'senas-v2-theme') applyTheme(e.newValue === 'light');
  });
})();