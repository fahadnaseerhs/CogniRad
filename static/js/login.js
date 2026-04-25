/* ============================================================
   COGNIRAD — login.js
   Responsibilities:
     1. Animated 3D wave background (WebGL via canvas)
     2. Form validation
     3. POST /auth/login
     4. sessionStorage write
     5. Transition to app.html
   ============================================================ */

'use strict';

/* ── WAVE BACKGROUND ─────────────────────────────────────── */
(function initWave() {
  const canvas = document.getElementById('wave-canvas');
  if (!canvas) return;

  const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

  // WebGL not supported — fallback to CSS animation, nothing breaks
  if (!gl) {
    canvas.style.display = 'none';
    return;
  }

  // Vertex shader: a fullscreen quad
  const vsSource = `
    attribute vec2 a_pos;
    void main() {
      gl_Position = vec4(a_pos, 0.0, 1.0);
    }
  `;

  // Fragment shader: layered sine waves with perspective depth
  const fsSource = `
    precision mediump float;
    uniform float u_time;
    uniform vec2  u_res;

    float wave(vec2 uv, float freq, float speed, float amp, float offset) {
      return amp * sin(uv.x * freq + u_time * speed + offset);
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / u_res;

      // 3D perspective warp: compress x toward center as y approaches horizon
      float horizon = 0.52;
      float depth = abs(uv.y - horizon);
      float perspective = 1.0 - smoothstep(0.0, 0.5, depth) * 0.7;
      vec2 warpedUV = vec2((uv.x - 0.5) / perspective + 0.5, uv.y);

      float brightness = 0.0;

      // Multiple wave layers at different depths
      for (int i = 1; i <= 6; i++) {
        float fi = float(i);
        float layerY = horizon + wave(warpedUV, 3.0 + fi * 1.2, 0.4 + fi * 0.08, 0.045 / fi, fi * 1.3);
        float dist = abs(uv.y - layerY);
        float intensity = (1.0 / fi) * 0.7;
        brightness += intensity * smoothstep(0.018, 0.0, dist);
      }

      // Soft grid lines receding to horizon
      float gridX = abs(sin(warpedUV.x * 18.0 * perspective)) * 0.025 * perspective;
      float gridY = abs(sin(uv.y * 22.0)) * 0.015;
      brightness += gridX + gridY;

      // Vignette
      float vig = smoothstep(0.9, 0.3, length(uv - 0.5) * 1.4);
      brightness *= vig;

      // Green phosphor color
      vec3 col = vec3(0.0, 1.0, 0.255) * brightness;
      gl_FragColor = vec4(col, brightness * 0.9);
    }
  `;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.warn('Shader error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  const vs = compileShader(gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) { canvas.style.display = 'none'; return; }

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    canvas.style.display = 'none'; return;
  }

  gl.useProgram(program);

  // Fullscreen quad
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1, -1,  1,
    -1,  1,  1, -1,  1,  1
  ]), gl.STATIC_DRAW);

  const posLoc  = gl.getAttribLocation(program, 'a_pos');
  const timeLoc = gl.getUniformLocation(program, 'u_time');
  const resLoc  = gl.getUniformLocation(program, 'u_res');

  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  // Enable alpha blending
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  let startTime = performance.now();

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  resize();
  window.addEventListener('resize', resize, { passive: true });

  function render() {
    const t = (performance.now() - startTime) / 1000;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform1f(timeLoc, t);
    gl.uniform2f(resLoc, canvas.width, canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(render);
  }
  render();
})();


/* ── FORM LOGIC ──────────────────────────────────────────── */
(function initLogin() {

  const API_BASE = window.COGNIRAD_API || 'http://localhost:8000';

  // DOM refs
  const form        = document.getElementById('login-form');
  const input       = document.getElementById('cms-input');
  const btn         = document.getElementById('login-btn');
  const btnText     = document.getElementById('btn-text');
  const errorMsg    = document.getElementById('error-msg');
  const loadingOv   = document.getElementById('loading-overlay');

  if (!form) return;

  /* ── Validation ──────────────────────────────────────── */
  function validate(value) {
    const trimmed = value.trim();
    if (!trimmed) return 'CMS ID is required';
    if (trimmed.length < 3) return 'CMS ID too short';
    // Allow formats: UID-000000, plain numbers, alphanumeric
    if (!/^[A-Za-z0-9\-_]+$/.test(trimmed)) return 'Invalid characters in CMS ID';
    return null;
  }

  function showError(msg) {
    errorMsg.textContent = '// ERR: ' + msg.toUpperCase();
    errorMsg.classList.add('visible');
  }

  function clearError() {
    errorMsg.classList.remove('visible');
  }

  /* ── Auth flow ───────────────────────────────────────── */
  async function handleSubmit(e) {
    e.preventDefault();
    clearError();

    const cms = input.value.trim().toUpperCase();
    const validationError = validate(cms);
    if (validationError) {
      showError(validationError);
      input.focus();
      return;
    }

    // Loading state
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ cms_id: cms })
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Auth failed (${res.status})`);
      }

      const data = await res.json();

      // Write everything app.html will need
      sessionStorage.setItem('cognirad_token',    data.token);
      sessionStorage.setItem('cognirad_cms',      cms);
      sessionStorage.setItem('cognirad_name',     data.student_name  || cms);
      sessionStorage.setItem('cognirad_channel',  data.channel_id    || '');
      sessionStorage.setItem('cognirad_freq',     data.channel_freq  || '');
      sessionStorage.setItem('cognirad_status',   data.channel_status || 'FREE');

      // Show loading transition then navigate
      transitionToApp();

    } catch (err) {
      setLoading(false);
      showError(err.message || 'Connection failed');
    }
  }

  /* ── Loading state ───────────────────────────────────── */
  function setLoading(on) {
    btn.classList.toggle('loading', on);
    btnText.textContent = on ? 'ESTABLISHING...' : 'ESTABLISH LINK';
    input.disabled = on;
  }

  /* ── Page transition ─────────────────────────────────── */
  function transitionToApp() {
    if (loadingOv) {
      loadingOv.classList.add('active');
      // Navigate after loading bar animation completes
      setTimeout(() => {
        window.location.href = '/static/app.html';
      }, 1900);
    } else {
      window.location.href = '/static/app.html';
    }
  }

  /* ── Clear error on type ─────────────────────────────── */
  input.addEventListener('input', clearError);

  /* ── Keyboard: Enter submits ─────────────────────────── */
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSubmit(e);
  });

  form.addEventListener('submit', handleSubmit);

  /* ── Auto-format CMS ID ──────────────────────────────── */
  input.addEventListener('input', function () {
    // Remove invalid chars as user types
    this.value = this.value.replace(/[^A-Za-z0-9\-_]/g, '').toUpperCase();
  });

  /* ── Focus input on load ─────────────────────────────── */
  setTimeout(() => input.focus(), 300);

})();
