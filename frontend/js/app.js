/* ============================================================
   frontend/js/app.js  v2.2
   ============================================================ */

const API_BASE = window.location.origin + '/api';
const WS_BASE  = (window.location.protocol === 'https:' ? 'wss://' : 'ws://')
                 + window.location.host;

// ============================================================
// AUTH
// ============================================================
const Auth = {
  guardar(token, usuario) {
    localStorage.setItem('token', token);
    localStorage.setItem('usuario', JSON.stringify(usuario));
  },
  token()   { return localStorage.getItem('token'); },
  usuario() { const u = localStorage.getItem('usuario'); return u ? JSON.parse(u) : null; },
  loggedIn(){ return !!this.token(); },
  esAdmin() { return this.usuario()?.rol === 'admin'; },
  logout()  { localStorage.clear(); window.location.href = '/index.html'; },
  requerir(){ if (!this.loggedIn()) window.location.href = '/index.html'; },
  redirigirSiLogueado() { if (this.loggedIn()) window.location.href = '/dashboard.html'; }
};

// ============================================================
// API HTTP
// ============================================================
const Api = {
  async request(path, opts = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...(Auth.token() ? { 'Authorization': `Bearer ${Auth.token()}` } : {}),
      ...(opts.headers || {}),
    };
    const res = await fetch(`${API_BASE}${path}`, {
      ...opts, headers,
      body: opts.body
        ? (typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body))
        : undefined,
    });
    if (res.status === 401) { Auth.logout(); return; }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
    return data;
  },
  get(path)         { return this.request(path); },
  post(path, body)  { return this.request(path, { method: 'POST',   body }); },
  patch(path, body) { return this.request(path, { method: 'PATCH',  body }); },
  delete(path)      { return this.request(path, { method: 'DELETE'       }); },
};

// ============================================================
// SELECTOR DE CÁMARA — menú desplegable
// ============================================================
const SelectorCamara = {
  async listar() {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(s => s.getTracks().forEach(t => t.stop())).catch(() => {});
      const devs = await navigator.mediaDevices.enumerateDevices();
      return devs.filter(d => d.kind === 'videoinput');
    } catch(e) { return []; }
  },

  _clasificar(label) {
    const l = label.toLowerCase();
    if (l.includes('droidcam') || l.includes('ivcam')    || l.includes('epoccam') ||
        l.includes('virtual')  || l.includes('obs')      || l.includes('wifi')    ||
        l.includes('wireless') || l.includes('network')  || l.includes('phone link') ||
        l.includes('mobile')   || l.includes('android')  || l.includes('iphone')  ||
        l.includes('continuity')|| l.includes('camo'))
      return { label: 'Red / Virtual', color: '#f59e0b' };
    if (l.includes('usb')    || l.includes('logitech') || l.includes('webcam') ||
        l.includes('hd pro') || l.includes('c920')     || l.includes('c922')   ||
        l.includes('external')|| l.includes('externa'))
      return { label: 'USB / Externa', color: '#38d9a9' };
    if (l.includes('integrated') || l.includes('integrada') || l.includes('built-in') ||
        l.includes('facetime')   || l.includes('front')     || l.includes('frontal')  ||
        l.includes('internal')   || l.includes('interna')   || l.includes('hd camera')||
        l.includes('laptop'))
      return { label: 'Integrada', color: '#4f8ef7' };
    return { label: 'Detectada', color: '#738099' };
  },

  mostrarMenu(anchorEl) {
    return new Promise(async (resolve) => {
      this._cerrar();
      const camaras = await this.listar();
      if (!camaras.length) { toast('No se detectaron cámaras', 'error'); resolve(null); return; }

      if (!document.getElementById('__cam-style__')) {
        const s = document.createElement('style');
        s.id = '__cam-style__';
        s.textContent = `
          @keyframes camIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
          .__ci:hover { background:#1c2535 !important; }
        `;
        document.head.appendChild(s);
      }

      const rect = anchorEl.getBoundingClientRect();
      const menuH = camaras.length * 60 + 70;
      const spaceAbove = rect.top;
      const top = spaceAbove > menuH ? rect.top - menuH - 6 : rect.bottom + 6;

      const menu = document.createElement('div');
      menu.id = '__cam-menu__';
      menu.style.cssText = `
        position:fixed; top:${top}px;
        left:${Math.min(rect.left, window.innerWidth - 300)}px;
        min-width:270px; max-width:340px;
        background:#151b25; border:1px solid #2a3550;
        border-radius:12px; box-shadow:0 8px 32px rgba(0,0,0,.65);
        z-index:99999; overflow:hidden; animation:camIn .15s ease;
      `;

      menu.innerHTML = `
        <div style="padding:.5rem .85rem;font-size:.68rem;font-weight:700;
                    color:#738099;text-transform:uppercase;letter-spacing:.07em;
                    border-bottom:1px solid #2a3550;
                    display:flex;justify-content:space-between">
          <span>Seleccionar cámara</span>
          <span style="color:#4f8ef7">${camaras.length} disponible${camaras.length>1?'s':''}</span>
        </div>
      `;

      camaras.forEach((cam, i) => {
        const name = cam.label || `Cámara ${i+1}`;
        const { label, color } = this._clasificar(name);
        const item = document.createElement('div');
        item.className = '__ci';
        item.style.cssText = `
          display:flex;align-items:center;gap:.65rem;padding:.55rem .85rem;
          cursor:pointer;transition:background .1s ease;border-bottom:1px solid #1a2030;
        `;
        item.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0">
            <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
          </svg>
          <div style="flex:1;min-width:0">
            <div style="font-size:.82rem;font-weight:600;color:#e8edf5;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${name}</div>
            <span style="font-size:.67rem;padding:.1rem .4rem;border-radius:999px;
                         background:${color}22;color:${color};border:1px solid ${color}44">${label}</span>
          </div>
        `;
        item.addEventListener('click', () => { this._cerrar(); resolve(cam.deviceId); });
        menu.appendChild(item);
      });

      const cancel = document.createElement('div');
      cancel.style.cssText = `padding:.4rem;text-align:center;font-size:.75rem;color:#738099;cursor:pointer;`;
      cancel.textContent = 'Cancelar';
      cancel.addEventListener('mouseenter', () => cancel.style.color='#e8edf5');
      cancel.addEventListener('mouseleave', () => cancel.style.color='#738099');
      cancel.addEventListener('click', () => { this._cerrar(); resolve(null); });
      menu.appendChild(cancel);

      document.body.appendChild(menu);
      setTimeout(() => {
        document.addEventListener('click', (e) => {
          if (!menu.contains(e.target)) { this._cerrar(); resolve(null); }
        }, { once: true });
      }, 80);
    });
  },

  _cerrar() { document.getElementById('__cam-menu__')?.remove(); }
};

// ============================================================
// WEBSOCKET
// ============================================================
class SalaWS {
  constructor(salaId, handlers = {}) {
    this.salaId=salaId; this.handlers=handlers;
    this.ws=null; this._intentos=0; this._timer=null;
  }
  conectar() {
    const url = `${WS_BASE}/ws/${this.salaId}?token=${Auth.token()}`;
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.ws.onopen    = () => { this._intentos=0; this.handlers.onConectado?.(); };
    this.ws.onmessage = (e) => { try { this._dispatch(JSON.parse(e.data)); } catch{} };
    this.ws.onerror   = (e) => this.handlers.onError?.(e);
    this.ws.onclose   = (e) => {
      this.handlers.onDesconectado?.(e);
      if (e.code!==1000 && this._intentos<5) {
        this._intentos++;
        this._timer = setTimeout(()=>this.conectar(), Math.min(1000*this._intentos,8000));
      }
    };
  }
  _dispatch(d) {
    const m = {
      mensaje_chat:'onMensaje', traduccion_live:'onTraduccion', sistema:'onSistema',
      sala_info:'onParticipantes', 'webrtc_señal':'onWebRTC',
      mensaje_corregido_ia:'onCorreccionIA', limpiar_subtitulos:'onLimpiar'
    };
    if (m[d.tipo]) this.handlers[m[d.tipo]]?.(d);
  }
  enviar(d)        { if(this.ws?.readyState===WebSocket.OPEN){this.ws.send(JSON.stringify(d));return true;}return false; }
  enviarFrame(b64) { return this.enviar({tipo:'frame',data:b64}); }
  enviarTexto(txt) { return this.enviar({tipo:'mensaje_texto',texto:txt}); }
  finFrase()       { return this.enviar({tipo:'fin_frase'}); }
  limpiar()        { return this.enviar({tipo:'limpiar'}); }
  señalWebRTC(id,s){ return this.enviar({tipo:'webrtc_señal',para:id,señal:s}); }
  desconectar()    { clearTimeout(this._timer); this._intentos=999; this.ws?.close(1000,'bye'); }
}

// ============================================================
// CAPTURADOR DE FRAMES
// ============================================================
class CapturadorCamara {
  constructor(videoEl, canvasEl, onFrame, fps=12) {
    this.video=videoEl; this.canvas=canvasEl;
    this.ctx=canvasEl.getContext('2d'); this.onFrame=onFrame;
    this.fps=fps; this._timer=null; this._stream=null; this.deviceId=null;
  }

  async iniciar(anchorEl) {
    const deviceId = await SelectorCamara.mostrarMenu(anchorEl);
    if (!deviceId) return false;
    return this._abrir(deviceId);
  }

  // switchCamara: cambia cámara en caliente
  async switchCamara(anchorEl) {
    clearInterval(this._timer);
    const deviceId = await SelectorCamara.mostrarMenu(anchorEl);
    if (!deviceId) {
      this._timer = setInterval(() => this._capturar(), 1000 / this.fps);
      return false;
    }
    return this._abrir(deviceId);
  }

  async _abrir(deviceId) {
    try {
      this._stream?.getTracks().forEach(t => t.stop());
      this._stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId:{exact:deviceId}, width:{ideal:640}, height:{ideal:480} },
        audio: false,
      });
      this.video.srcObject = this._stream;
      await new Promise(res => { this.video.onloadedmetadata=res; setTimeout(res,2000); });
      await this.video.play().catch(()=>{});
      this.canvas.width=640; this.canvas.height=480; this.deviceId=deviceId;
      const label = this._stream.getVideoTracks()[0]?.label || 'Cámara';
      console.log('[Camara] Activa:', label);
      toast(`Cámara: ${label}`, 'success');
      clearInterval(this._timer);
      this._timer = setInterval(() => this._capturar(), 1000/this.fps);
      return true;
    } catch(e) {
      console.error('[Camara]', e);
      toast('No se pudo abrir la cámara: ' + e.message, 'error');
      return false;
    }
  }

  _capturar() {
    if (!this.video.videoWidth) return;
    this.ctx.save(); this.ctx.scale(-1,1);
    this.ctx.drawImage(this.video,-this.canvas.width,0,this.canvas.width,this.canvas.height);
    this.ctx.restore();
    this.onFrame(this.canvas.toDataURL('image/jpeg',0.7));
  }

  detener() {
    clearInterval(this._timer);
    this._stream?.getTracks().forEach(t=>t.stop());
    this.video.srcObject=null; this._stream=null; this.deviceId=null;
  }
}

// ============================================================
// WEBRTC
// ============================================================
class GestorWebRTC {
  constructor(sala,miId){this.sala=sala;this.miId=miId;this.peers={};this._local=null;}
  async iniciarStream(){
    try{this._local=await navigator.mediaDevices.getUserMedia({video:true,audio:true});return this._local;}
    catch(e){return null;}
  }
  async llamar(id){
    const pc=this._peer(id),o=await pc.createOffer();
    await pc.setLocalDescription(o);this.sala.señalWebRTC(id,{type:'offer',sdp:o.sdp});
  }
  async manejarSeñal(deId,s){
    if(!this.peers[deId])this._peer(deId);const pc=this.peers[deId];
    if(s.type==='offer'){await pc.setRemoteDescription(new RTCSessionDescription(s));const a=await pc.createAnswer();await pc.setLocalDescription(a);this.sala.señalWebRTC(deId,{type:'answer',sdp:a.sdp});}
    else if(s.type==='answer')await pc.setRemoteDescription(new RTCSessionDescription(s));
    else if(s.type==='ice')await pc.addIceCandidate(new RTCIceCandidate(s.candidate));
  }
  _peer(id){
    const pc=new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
    this._local?.getTracks().forEach(t=>pc.addTrack(t,this._local));
    pc.onicecandidate=(e)=>{if(e.candidate)this.sala.señalWebRTC(id,{type:'ice',candidate:e.candidate});};
    pc.ontrack=(e)=>document.dispatchEvent(new CustomEvent('webrtc:stream',{detail:{usuarioId:id,stream:e.streams[0]}}));
    this.peers[id]=pc;return pc;
  }
  colgar(){Object.values(this.peers).forEach(p=>p.close());this.peers={};this._local?.getTracks().forEach(t=>t.stop());this._local=null;}
}

// ============================================================
// TOASTS
// ============================================================
function toast(msg, tipo='info', dur=3500) {
  let c=document.getElementById('toast-container');
  if(!c){c=document.createElement('div');c.id='toast-container';document.body.appendChild(c);}
  const icons={success:'✓',error:'✕',info:'ℹ'};
  const el=document.createElement('div');
  el.className=`toast ${tipo}`;
  el.innerHTML=`<span style="font-weight:700">${icons[tipo]||'ℹ'}</span> ${msg}`;
  c.appendChild(el);
  setTimeout(()=>{el.style.opacity='0';el.style.transform='translateX(120%)';el.style.transition='300ms ease';setTimeout(()=>el.remove(),300);},dur);
}

function formatHora(iso){return new Date(iso).toLocaleTimeString('es',{hour:'2-digit',minute:'2-digit'});}
function formatFecha(iso){return new Date(iso).toLocaleDateString('es',{day:'2-digit',month:'short',year:'numeric'});}
function iniciales(n){return(n||'?').split(' ').slice(0,2).map(p=>p[0]).join('').toUpperCase();}

window.Auth=Auth;window.Api=Api;window.SelectorCamara=SelectorCamara;
window.SalaWS=SalaWS;window.CapturadorCamara=CapturadorCamara;window.GestorWebRTC=GestorWebRTC;
window.toast=toast;window.formatHora=formatHora;window.formatFecha=formatFecha;window.iniciales=iniciales;