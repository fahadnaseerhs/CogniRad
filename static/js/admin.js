// CogniRad Admin Dashboard — Spectrum Control v10
// ═══════════════════════════════════════════════════════════════════════════
// CogniRad Admin Dashboard — Spectrum Control  v10
// ═══════════════════════════════════════════════════════════════════════════

const API_BASE = window.location.origin;
const WS_BASE = API_BASE.replace(/^http/, 'ws');

const CHANNELS = ['CH-1', 'CH-2', 'CH-3', 'CH-4', 'CH-5'];
const CH_COLORS = {
  'CH-1': '#67a8ff',
  'CH-2': '#52d7ff',
  'CH-3': '#f4bd50',
  'CH-4': '#ff5e67',
  'CH-5': '#b48cff',
};

// 5G Network Slicing labels for presentation
const SLICE_NAMES = {
  'CH-1': 'PU — eMBB Slice',
  'CH-2': 'SU — URLLC Slice',
  'CH-3': 'SU — mMTC Slice',
  'CH-4': 'V2X Edge Slice',
  'CH-5': 'Public Safety',
};
const SLICE_SHORT = {
  'CH-1': 'eMBB',
  'CH-2': 'URLLC',
  'CH-3': 'mMTC',
  'CH-4': 'V2X',
  'CH-5': 'Safety',
};

// sqrt(N) dynamic jammed ceiling coefficient (matches classifier.py _DELTA=35)
const JAMMED_COEFF = 35.0;

// ── State ──────────────────────────────────────────────────────────────────
let adminKey = sessionStorage.getItem('cognirad_admin_key') || '';
let channels = {};   // CH-x → snapshot from SPECTRUM_TICK
let students = [];   // from /admin/students
let spectrumSocket = null;
let pollTimer = null;
let totalReallocations = 0;

// Rolling history — 60 points per channel per metric
const HISTORY_LEN = 60;
const history = {};
CHANNELS.forEach(ch => {
  history[ch] = { energy: [], snr: [], slope: [], members: [] };
});
let chartLabels = [];

// ── DOM refs ───────────────────────────────────────────────────────────────
const loginView = document.getElementById('login-view');
const adminApp = document.getElementById('admin-app');
const loginForm = document.getElementById('admin-login');
const loginError = document.getElementById('login-error');
const passwordInput = document.getElementById('admin-password');
const channelGrid = document.getElementById('channel-grid');
const studentRows = document.getElementById('student-rows');
const studentSearch = document.getElementById('student-search');
const eventLog = document.getElementById('event-log');
const msgFeed = document.getElementById('msg-feed');
const stuEnergyList = document.getElementById('student-energy-list');
const feedBadge = document.getElementById('feed-badge');
const feedLabel = document.getElementById('feed-label');

// ── Helpers ────────────────────────────────────────────────────────────────
function fmt(v, d = 2) { return Number(v || 0).toFixed(d); }
function nowLabel() {
  return new Date().toLocaleTimeString([], { hour12: false });
}

// Dynamic jammed ceiling for N users (mirrors classifier.py)
function jammedCeiling(n) {
  return JAMMED_COEFF * Math.sqrt(Math.max(n, 1));
}

function statusClass(s) {
  return { FREE: 'FREE', BUSY: 'BUSY', CONGESTED: 'CONGESTED', JAMMED: 'JAMMED' }[s] || 'FREE';
}

function deliveryClass(d) {
  if (!d) return 'ok';
  if (d.includes('REJECT')) return 'err';
  if (d.includes('DEGRADED')) return 'warn';
  if (d.includes('STABILIZATION')) return 'info';
  if (d.includes('OFFLINE')) return 'muted';
  return 'ok';
}

function deliveryLabel(d) {
  if (!d) return 'DELIVERED';
  return d.replace(/_/g, ' ');
}

// ── Clock ──────────────────────────────────────────────────────────────────
function startClock() {
  const el = document.getElementById('sys-clock');
  function tick() {
    if (el) el.textContent = new Date().toLocaleTimeString([], { hour12: false });
  }
  tick();
  setInterval(tick, 1000);
}

// ── Feed badge ─────────────────────────────────────────────────────────────
function setFeedState(state, text) {
  if (!feedBadge || !feedLabel) return;
  feedBadge.className = 'feed-badge ' + state;
  feedLabel.textContent = text;
}

// ── Event log ──────────────────────────────────────────────────────────────
function logEvent(text, type = '') {
  if (!eventLog) return;
  const row = document.createElement('div');
  row.className = 'event-row';
  row.innerHTML = `<div class="event-time">${nowLabel()}</div><div class="event-body">${text}</div>`;
  eventLog.prepend(row);
  while (eventLog.children.length > 100) eventLog.lastChild.remove();
}

// ── Message feed ───────────────────────────────────────────────────────────
function addMsgRow({ time, sender, recipient, channel, energy, delivery, bitrate_bps, utilization, modulation }) {
  if (!msgFeed) return;
  const dc = deliveryClass(delivery);
  const lbl = deliveryLabel(delivery);

  let bpsStr = '';
  if (bitrate_bps > 0) {
    bpsStr = bitrate_bps >= 1e6
      ? `${(bitrate_bps / 1e6).toFixed(1)} Mbps`
      : `${(bitrate_bps / 1e3).toFixed(0)} kbps`;
  }
  const utilStr = utilization > 0 ? `${(utilization * 100).toFixed(1)}% util` : '';

  const row = document.createElement('div');
  const rowClass = dc === 'err' ? 'rejected' : dc === 'warn' ? 'degraded' : dc === 'info' ? 'stabilized' : dc === 'muted' ? 'offline' : 'delivered';
  row.className = `msg-row ${rowClass}`;
  row.innerHTML = `
    <div class="msg-top">
      <span class="msg-time">${time || nowLabel()}</span>
      <span class="msg-sender">${sender}</span>
      <span class="msg-arrow">→</span>
      <span class="msg-recip">${recipient}</span>
      <span class="msg-ch">${channel}</span>
      <span class="msg-delivery ${dc}">${lbl}</span>
    </div>
    ${(bpsStr || utilStr || modulation) ? `<div class="msg-phy">${[bpsStr, utilStr, modulation, energy ? fmt(energy) + ' J' : ''].filter(Boolean).join('  ·  ')}</div>` : ''}
  `;
  msgFeed.prepend(row);
  while (msgFeed.children.length > 20) msgFeed.lastChild.remove();
}

// ── Admin fetch ────────────────────────────────────────────────────────────
async function adminFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function verifyAdmin(key) {
  const k = key.trim();
  if (!k) throw new Error('Password cannot be empty.');
  await adminFetch('/admin/verify', {
    method: 'POST',
    body: JSON.stringify({ admin_key: k }),
  });
  return k;
}

// ── Auth flow ──────────────────────────────────────────────────────────────
function unlock() {
  loginView.style.display = 'none';
  adminApp.hidden = false;
  adminApp.style.display = 'flex';
  startClock();
  buildLegends();
  refreshAll().then(() => {
    drawAllCharts();
    startPolling();
    connectSpectrum();
    logEvent('Admin session unlocked.');
  }).catch(err => {
    setFeedState('error', 'DATA ERROR');
    logEvent(`Data load failed: ${err.message}`);
  });
}

function lock() {
  sessionStorage.removeItem('cognirad_admin_key');
  adminKey = '';
  if (spectrumSocket) spectrumSocket.close();
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  loginView.style.display = 'grid';
  adminApp.hidden = true;
  adminApp.style.display = 'none';
  if (passwordInput) { passwordInput.value = ''; passwordInput.focus(); }
}

loginForm.addEventListener('submit', async e => {
  e.preventDefault();
  const key = passwordInput.value.trim();
  if (!key) { loginError.textContent = 'Password is required.'; return; }
  loginError.textContent = 'Verifying…';
  try {
    adminKey = await verifyAdmin(key);
    sessionStorage.setItem('cognirad_admin_key', adminKey);
    loginError.textContent = '';
    unlock();
  } catch (err) {
    loginError.textContent = err.message || 'Invalid password.';
    passwordInput.select();
  }
});

// Show/hide password
const togglePw = document.getElementById('toggle-pw');
if (togglePw && passwordInput) {
  togglePw.addEventListener('click', () => {
    passwordInput.type = passwordInput.type === 'password' ? 'text' : 'password';
    togglePw.textContent = passwordInput.type === 'password' ? '👁' : '🙈';
    passwordInput.focus();
  });
}

document.getElementById('lock-btn').addEventListener('click', lock);
document.getElementById('refresh-btn').addEventListener('click', () => refreshAll().then(drawAllCharts));
document.getElementById('clear-log-btn').addEventListener('click', () => { if (eventLog) eventLog.innerHTML = ''; });
document.getElementById('clear-feed-btn').addEventListener('click', () => { if (msgFeed) msgFeed.innerHTML = ''; });
studentSearch.addEventListener('input', renderStudents);

document.getElementById('simulate-btn').addEventListener('click', async () => {
  const energy = Number(document.getElementById('energy-input').value || 5);
  try {
    await adminFetch('/admin/simulate_load', {
      method: 'POST',
      body: JSON.stringify({ admin_key: adminKey, energy_per_student: energy }),
    });
    logEvent(`Simulation: ${fmt(energy, 1)} J per student applied.`);
    await refreshAll();
    drawAllCharts();
  } catch (err) {
    logEvent(`Simulation failed: ${err.message}`);
  }
});

// ── Data loading ───────────────────────────────────────────────────────────
async function refreshAll() {
  await Promise.all([loadChannels(), loadStudents()]);
}

async function loadChannels() {
  const data = await adminFetch('/channel/state');
  channels = data.channels || {};
  renderChannels();
  updateMetrics();
  pushSnapshotToHistory(channels);
  drawAllCharts();
}

async function loadStudents() {
  const data = await adminFetch(`/admin/students?admin_key=${encodeURIComponent(adminKey)}`);
  students = data.students || [];
  renderStudents();
  renderStudentEnergyList();
  updateMetrics();
}

// ── Metrics ────────────────────────────────────────────────────────────────
function updateMetrics() {
  const totalEnergy = Object.values(channels).reduce((s, c) => s + Number(c.total_energy || 0), 0);
  const jammedCount = Object.values(channels).filter(c => c.status === 'JAMMED').length;
  document.getElementById('metric-channels').textContent = Object.keys(channels).length || 5;
  document.getElementById('metric-students').textContent = students.length;
  document.getElementById('metric-online').textContent = students.filter(s => s.is_online).length;
  document.getElementById('metric-energy').textContent = `${fmt(totalEnergy)} J`;
  document.getElementById('metric-jammed').textContent = jammedCount;
  document.getElementById('metric-reallocations').textContent = totalReallocations;
}

// ── Channel cards ──────────────────────────────────────────────────────────
function renderChannels() {
  channelGrid.innerHTML = CHANNELS.map(key => {
    const ch = channels[key] || {};
    const status = ch.status || 'FREE';
    const energy = Number(ch.total_energy || 0);
    const n = ch.user_count ?? (ch.users || []).length ?? 0;
    const ceil = jammedCeiling(n);
    const pct = Math.min(100, (energy / ceil) * 100);
    const color = CH_COLORS[key];
    const isJammed = status === 'JAMMED';

    return `
      <article class="channel-card status-${status}" style="--ch-color:${color}" data-ch="${key}">
        <div class="channel-top">
          <div class="channel-name">${SLICE_NAMES[key] || key}<span class="ch-key-sub">${key}</span></div>
          <div class="status-pill ${statusClass(status)}">${status}</div>
        </div>
        <div class="channel-freq">${ch.frequency || '--'}</div>
        <div class="ebar"><div class="ebar-fill" style="width:${pct}%"></div></div>
        <div class="channel-stats">
          <div class="stat-row">
            <span class="stat-label">Energy</span>
            <span class="stat-val">${fmt(energy)} J</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">Users</span>
            <span class="stat-val">${n}</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">SNR</span>
            <span class="stat-val">${fmt(ch.snr_db, 1)} dB</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">Mod</span>
            <span class="stat-val">${ch.modulation || '--'}</span>
          </div>
        </div>
        <div class="channel-actions">
          <button type="button" class="btn-jam ${isJammed ? 'active-jam' : ''}"
                  data-action="jam" data-channel="${key}">
            ${isJammed ? '⚡ JAMMED' : 'Jam'}
          </button>
          <button type="button" class="btn-unjam"
                  data-action="unjam" data-channel="${key}">Unjam</button>
          <button type="button" class="btn-reallocate"
                  data-action="reallocate" data-channel="${key}">Reallocate</button>
        </div>
      </article>
    `;
  }).join('');

  channelGrid.querySelectorAll('button[data-action]').forEach(btn => {
    btn.addEventListener('click', () => runChannelAction(btn.dataset.action, btn.dataset.channel));
  });
}

async function runChannelAction(action, channelKey) {
  const endpoint = action === 'jam' ? '/admin/jam'
    : action === 'unjam' ? '/admin/unjam'
      : '/admin/reallocate';

  try {
    const result = await adminFetch(endpoint, {
      method: 'POST',
      body: JSON.stringify({ admin_key: adminKey, channel_key: channelKey }),
    });

    if (action === 'jam') {
      const moved = result.users_moved || [];
      logEvent(`⚡ ${channelKey} FORCE JAMMED — ${moved.length} student(s) evacuated.`);
      // Add to message feed as a system event
      addMsgRow({
        time: nowLabel(),
        sender: 'ADMIN',
        recipient: channelKey,
        channel: channelKey,
        energy: 0,
        delivery: 'CHANNEL_JAMMED',
        bitrate_bps: 0,
        utilization: 0,
        modulation: '',
      });
    } else if (action === 'unjam') {
      logEvent(`✓ ${channelKey} unjammed — channel restored to FREE.`);
    } else {
      const moved = result.moved || [];
      logEvent(`↺ ${channelKey} reallocated — ${moved.length} student(s) moved.`);
      totalReallocations += moved.length;
    }

    await refreshAll();
    drawAllCharts();
  } catch (err) {
    logEvent(`✗ ${action} ${channelKey} failed: ${err.message}`);
  }
}

// ── Students table ─────────────────────────────────────────────────────────
function renderStudents() {
  const q = (studentSearch.value || '').trim().toLowerCase();
  const filtered = students.filter(s =>
    !q || `${s.cms} ${s.name || ''} ${s.channel_key || ''}`.toLowerCase().includes(q)
  );
  studentRows.innerHTML = filtered.map(s => `
    <tr>
      <td><span class="online-dot ${s.is_online ? 'on' : ''}"></span>${s.cms}</td>
      <td>${s.name || '--'}</td>
      <td>${s.channel_key || '--'}</td>
      <td>${fmt(s.energy)}</td>
      <td>${s.is_online ? '<span class="FREE">ONLINE</span>' : '<span style="color:var(--muted)">OFFLINE</span>'}</td>
    </tr>
  `).join('');
}

// ── Student energy bars (terminal-style) ───────────────────────────────────
function renderStudentEnergyList() {
  if (!stuEnergyList) return;

  // Show online students first, then offline, sorted by energy desc
  const sorted = [...students].sort((a, b) => {
    if (a.is_online !== b.is_online) return a.is_online ? -1 : 1;
    return Number(b.energy || 0) - Number(a.energy || 0);
  });

  if (!sorted.length) {
    stuEnergyList.innerHTML = '<div class="stu-empty">No students registered.</div>';
    return;
  }

  // Find max energy for bar scaling
  const maxE = Math.max(...sorted.map(s => Number(s.energy || 0)), 1);

  stuEnergyList.innerHTML = sorted.map(s => {
    const e = Number(s.energy || 0);
    const pct = Math.min(100, (e / maxE) * 100);
    // Color: green < 50%, yellow < 80%, red >= 80%
    const barColor = pct >= 80 ? 'var(--congested)' : pct >= 50 ? 'var(--busy)' : 'var(--ok)';
    const chColor = s.channel_key ? CH_COLORS[s.channel_key] || 'var(--muted)' : 'var(--muted)';

    return `
      <div class="stu-row">
        <div class="stu-dot ${s.is_online ? 'online' : ''}"></div>
        <div class="stu-name">${s.cms}</div>
        <div class="stu-ch" style="color:${chColor}">${s.channel_key || '—'}</div>
        <div class="stu-bar-wrap">
          <div class="stu-bar-fill" style="width:${pct}%;background:${barColor}"></div>
        </div>
        <div class="stu-energy">${fmt(e)} J</div>
      </div>
    `;
  }).join('');
}

// ── WebSocket — spectrum feed ──────────────────────────────────────────────
function connectSpectrum() {
  if (spectrumSocket) spectrumSocket.close();
  spectrumSocket = new WebSocket(`${WS_BASE}/ws/spectrum`);

  spectrumSocket.onopen = () => {
    setFeedState('live', 'LIVE');
    logEvent('Spectrum WebSocket connected.');
  };

  spectrumSocket.onclose = () => {
    setFeedState('polling', 'POLLING');
  };

  spectrumSocket.onerror = () => {
    setFeedState('error', 'WS ERROR');
  };

  spectrumSocket.onmessage = e => {
    let payload;
    try { payload = JSON.parse(e.data); } catch { return; }

    if (payload.type === 'SPECTRUM_HISTORY') {
      // Seed rolling history with past data
      CHANNELS.forEach(ch => {
        const pts = payload.channels[ch] || [];
        history[ch].energy = pts.slice(-HISTORY_LEN);
        // Pad labels
      });
      const recent = payload.recent_events || [];
      recent.forEach(ev => addReallocChip(ev));
      return;
    }

    if (payload.type !== 'SPECTRUM_TICK') return;

    // Update channel snapshots from tick
    CHANNELS.forEach(ch => {
      const row = payload.channels[ch];
      if (!row) return;
      // Merge into channels state
      if (!channels[ch]) channels[ch] = {};
      channels[ch].status = row.status;
      channels[ch].total_energy = row.energy;
      channels[ch].snr_db = row.snr_db;
      channels[ch].modulation = row.modulation;
      channels[ch].user_count = row.member_count;
    });

    // Push to history
    pushTickToHistory(payload);

    // Re-render channel cards
    renderChannels();
    updateMetrics();
    drawAllCharts();

    // Handle reallocation events
    const events = payload.events || [];
    events.forEach(ev => {
      if (ev.type === 'REALLOCATION') {
        totalReallocations++;
        logEvent(`↺ ${ev.user} moved ${ev.from} → ${ev.to}`);
        addReallocChip(ev);
        // Also show in message feed
        addMsgRow({
          time: nowLabel(),
          sender: ev.user,
          recipient: ev.to,
          channel: ev.from,
          energy: 0,
          delivery: 'REALLOCATED',
          bitrate_bps: 0,
          utilization: 0,
          modulation: '',
        });
      }
    });

    updateMetrics();
  };
}

// ── Polling fallback ───────────────────────────────────────────────────────
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      // Always poll students for energy updates
      await loadStudents();
      // Only poll channels if WS is not live
      if (!spectrumSocket || spectrumSocket.readyState !== WebSocket.OPEN) {
        await loadChannels();
        setFeedState('polling', 'POLLING');
      }
    } catch (err) {
      setFeedState('error', 'POLL ERR');
    }
  }, 2000);
}

// ── History management ─────────────────────────────────────────────────────
function pushPoint(arr, val) {
  arr.push(Number(val || 0));
  if (arr.length > HISTORY_LEN) arr.shift();
}

function pushTickToHistory(payload) {
  const t = new Date(payload.timestamp * 1000).toLocaleTimeString([], { hour12: false });
  chartLabels.push(t);
  if (chartLabels.length > HISTORY_LEN) chartLabels.shift();

  CHANNELS.forEach(ch => {
    const row = payload.channels[ch];
    if (!row) return;
    pushPoint(history[ch].energy, row.energy);
    pushPoint(history[ch].snr, row.snr_db);
    pushPoint(history[ch].slope, row.slope);
    pushPoint(history[ch].members, row.member_count);
  });
}

function pushSnapshotToHistory(snapshot) {
  chartLabels.push(nowLabel());
  if (chartLabels.length > HISTORY_LEN) chartLabels.shift();

  CHANNELS.forEach(ch => {
    const row = snapshot[ch];
    if (!row) return;
    const prev = history[ch].energy.length
      ? history[ch].energy[history[ch].energy.length - 1]
      : Number(row.total_energy || 0);
    pushPoint(history[ch].energy, row.total_energy);
    pushPoint(history[ch].snr, row.snr_db);
    pushPoint(history[ch].slope, Number(row.total_energy || 0) - prev);
    pushPoint(history[ch].members, row.user_count || 0);
  });
}

// ── Reallocation chips ─────────────────────────────────────────────────────
function addReallocChip(ev) {
  const container = document.getElementById('realloc-events');
  if (!container) return;
  const chip = document.createElement('div');
  chip.className = 'realloc-chip';
  chip.textContent = `${ev.user} ${ev.from}→${ev.to}`;
  container.prepend(chip);
  while (container.children.length > 30) container.lastChild.remove();
}

// ── Chart legends ──────────────────────────────────────────────────────────
function buildLegends() {
  ['energy-legend', 'snr-legend', 'slope-legend', 'member-legend', 'radar-legend'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = CHANNELS.map(ch => `
      <div class="legend-item">
        <div class="legend-dot" style="background:${CH_COLORS[ch]}"></div>
        ${SLICE_SHORT[ch] || ch}
      </div>
    `).join('');
  });
}

// ── Canvas chart renderer ──────────────────────────────────────────────────
function drawAllCharts() {
  drawLineChart('energy-chart', 'energy', 'J', null, null);
  drawLineChart('snr-chart', 'snr', 'dB', 0, 32);
  drawLineChart('slope-chart', 'slope', 'J/s', null, null);
  drawLineChart('member-chart', 'members', 'students', 0, null);
  drawRadarChart();
}

function drawLineChart(canvasId, key, unit, fixedMin = null, fixedMax = null) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const pad = { left: 48, right: 12, top: 14, bottom: 28 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Background
  ctx.fillStyle = '#07090d';
  ctx.fillRect(0, 0, W, H);

  // Collect all values for auto-scaling
  const allVals = CHANNELS.flatMap(ch => history[ch][key]);
  if (!allVals.length) {
    ctx.fillStyle = '#4a5568';
    ctx.font = '12px monospace';
    ctx.fillText('Waiting for data…', pad.left + 8, H / 2);
    return;
  }

  let minV = fixedMin !== null ? fixedMin : Math.min(...allVals);
  let maxV = fixedMax !== null ? fixedMax : Math.max(...allVals);
  if (minV === maxV) maxV = minV + 1;
  const span = maxV - minV;

  // Grid lines + Y labels
  const gridLines = 5;
  ctx.font = `11px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'right';
  for (let i = 0; i <= gridLines; i++) {
    const y = pad.top + (plotH / gridLines) * i;
    const val = maxV - (span / gridLines) * i;
    ctx.strokeStyle = '#1e2733';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = '#4a5568';
    ctx.fillText(
      key === 'members' ? Math.round(val).toString() : val.toFixed(1),
      pad.left - 4,
      y + 4
    );
  }

  // Zero line for slope chart
  if (key === 'slope' && minV < 0 && maxV > 0) {
    const zeroY = pad.top + plotH - ((0 - minV) / span) * plotH;
    ctx.strokeStyle = '#2b3340';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, zeroY);
    ctx.lineTo(W - pad.right, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Helper: compute XY points for a series
  function seriesPoints(series) {
    return series.map((v, i) => ({
      x: pad.left + (series.length === 1 ? 0 : (plotW * i) / (series.length - 1)),
      y: pad.top + plotH - ((v - minV) / span) * plotH,
    }));
  }

  // Draw each channel line with bezier smoothing + neon glow
  CHANNELS.forEach(ch => {
    const series = history[ch][key];
    if (!series.length) return;

    const color = CH_COLORS[ch];
    const pts = seriesPoints(series);

    // ── Gradient fill under curve ──
    ctx.beginPath();
    if (pts.length === 1) {
      ctx.moveTo(pts[0].x, pts[0].y);
    } else {
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        const cpx = (pts[i - 1].x + pts[i].x) / 2;
        ctx.bezierCurveTo(cpx, pts[i - 1].y, cpx, pts[i].y, pts[i].x, pts[i].y);
      }
    }
    const lastPt = pts[pts.length - 1];
    ctx.lineTo(lastPt.x, pad.top + plotH);
    ctx.lineTo(pts[0].x, pad.top + plotH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, color + '30');
    grad.addColorStop(1, color + '02');
    ctx.fillStyle = grad;
    ctx.fill();

    // ── Neon glow line ──
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    if (pts.length === 1) {
      ctx.moveTo(pts[0].x, pts[0].y);
    } else {
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        const cpx = (pts[i - 1].x + pts[i].x) / 2;
        ctx.bezierCurveTo(cpx, pts[i - 1].y, cpx, pts[i].y, pts[i].x, pts[i].y);
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.restore();

    // Dot at latest point
    ctx.beginPath();
    ctx.arc(lastPt.x, lastPt.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#07090d';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // X axis — latest timestamp label
  ctx.textAlign = 'left';
  ctx.fillStyle = '#4a5568';
  ctx.font = '10px monospace';
  const lastLabel = chartLabels[chartLabels.length - 1] || '';
  ctx.fillText(lastLabel, pad.left, H - 6);

  // Unit label top-right
  ctx.textAlign = 'right';
  ctx.fillStyle = '#2b3340';
  ctx.font = '10px monospace';
  ctx.fillText(unit, W - pad.right, pad.top - 2);
}

// ── Radar / Spider chart ───────────────────────────────────────────────────
function drawRadarChart() {
  const canvas = document.getElementById('radar-chart');
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(W, H) / 2 - 40;

  // Background
  ctx.fillStyle = '#07090d';
  ctx.fillRect(0, 0, W, H);

  const n = CHANNELS.length;
  const angleStep = (Math.PI * 2) / n;
  const startAngle = -Math.PI / 2; // top

  // Get max energy for normalization
  const energies = CHANNELS.map(ch => {
    const e = history[ch].energy;
    return e.length ? e[e.length - 1] : 0;
  });
  const maxE = Math.max(...energies, 1);

  // Draw concentric rings (20%, 40%, 60%, 80%, 100%)
  for (let ring = 1; ring <= 5; ring++) {
    const r = (R * ring) / 5;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const angle = startAngle + angleStep * (i % n);
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = ring === 5 ? '#2b3340' : '#161b24';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Percentage label
    ctx.fillStyle = '#2b3340';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${ring * 20}%`, cx + 3, cy - r + 10);
  }

  // Draw axis spokes + labels
  CHANNELS.forEach((ch, i) => {
    const angle = startAngle + angleStep * i;
    const xEnd = cx + R * Math.cos(angle);
    const yEnd = cy + R * Math.sin(angle);

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(xEnd, yEnd);
    ctx.strokeStyle = '#1e2733';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label
    const labelR = R + 20;
    const lx = cx + labelR * Math.cos(angle);
    const ly = cy + labelR * Math.sin(angle);
    ctx.fillStyle = CH_COLORS[ch];
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(SLICE_SHORT[ch] || ch, lx, ly);
  });

  // Draw filled polygon for current energy
  const normalized = energies.map(e => e / maxE);

  ctx.beginPath();
  normalized.forEach((val, i) => {
    const angle = startAngle + angleStep * i;
    const r = R * Math.max(val, 0.02); // min visible
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();

  // Gradient fill
  const rGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  rGrad.addColorStop(0, 'rgba(103,168,255,0.08)');
  rGrad.addColorStop(1, 'rgba(103,168,255,0.25)');
  ctx.fillStyle = rGrad;
  ctx.fill();

  // Glowing border
  ctx.save();
  ctx.shadowColor = '#67a8ff';
  ctx.shadowBlur = 12;
  ctx.strokeStyle = 'rgba(103,168,255,0.7)';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();

  // Dots at each vertex
  normalized.forEach((val, i) => {
    const angle = startAngle + angleStep * i;
    const r = R * Math.max(val, 0.02);
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    const color = CH_COLORS[CHANNELS[i]];

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#07090d';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

// ── Boot ───────────────────────────────────────────────────────────────────
// Auto-unlock if we already have a stored key
if (adminKey) {
  verifyAdmin(adminKey)
    .then(k => {
      adminKey = k;
      sessionStorage.setItem('cognirad_admin_key', k);
      unlock();
    })
    .catch(() => lock());
} else {
  lock();
}
