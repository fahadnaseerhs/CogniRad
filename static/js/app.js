// static/js/app.js — CogniRad Frontend
// ─────────────────────────────────────────────────────────────────────────
// Bugs fixed in this version:
//  1. DM messages now append directly to the DOM instead of full re-render,
//     so messages are always visible and scroll position is preserved.
//  2. Sent messages (MESSAGE_RESULT) also append directly — no flicker.
//  3. In-chat notification bubble no longer conflicts with renderMessages.
//  4. messages[cms] stores ALL messages for a conversation keyed by the
//     OTHER person's CMS, regardless of who sent it.
//  5. Filter state (same-channel / cross-channel / online) added.
//  6. Contacts sorted by most recent activity (last message timestamp).
// ─────────────────────────────────────────────────────────────────────────

const API_BASE = window.location.origin;
const WS_BASE = API_BASE.replace(/^http/, 'ws');
const token = sessionStorage.getItem('cognirad_token');
const myCms = sessionStorage.getItem('cognirad_cms');
const myName = sessionStorage.getItem('cognirad_name');
const INIT_FETCH_TIMEOUT_MS = 12000;

// ── DOM refs ──────────────────────────────────────────────────────────────
const loader = document.getElementById('app-loader');
const contactsList = document.getElementById('contacts-list');
const searchInput = document.getElementById('contacts-search');
const searchClear = document.getElementById('search-clear');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('chat-send-btn');
const backBtn = document.getElementById('chat-back-btn');
const notifBanner = document.getElementById('notif-banner');
const notifTitle = document.getElementById('notif-title');
const notifBody = document.getElementById('notif-body');

// ── State ─────────────────────────────────────────────────────────────────
let allStudents = [];   // all students except me
let channelsState = {};   // channel_key → channel data
let activeChatCms = null; // CMS of the person whose chat is open
let messages = {};   // otherCms → [msg objects]  (both sent & received)
let lastActivity = {};   // otherCms → epoch ms of last message
let unreadCounts = {};   // otherCms → int
let ws = null;
let myChannelKey = null;
let searchQuery = '';
let activeFilter = 'all'; // 'all' | 'same' | 'other' | 'online'
let isJamLocked = false;
let bootCompleted = false;
let bootWatchdogId = null;

async function fetchJsonWithTimeout(url, options = {}, timeoutMs = INIT_FETCH_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        if (!response.ok) {
            throw new Error(`${url} failed: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error(`${url} timed out after ${timeoutMs}ms`);
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}

function finishBoot() {
    if (bootCompleted) return;
    bootCompleted = true;
    if (bootWatchdogId) {
        clearTimeout(bootWatchdogId);
        bootWatchdogId = null;
    }
    if (!loader) return;
    loader.style.transition = 'opacity 0.4s ease';
    loader.style.opacity = '0';
    const hideLoader = () => { loader.style.display = 'none'; };
    loader.addEventListener('transitionend', hideLoader, { once: true });
    setTimeout(hideLoader, 600);
}

function failBoot(message) {
    if (!loader) return;
    loader.style.opacity = '1';
    loader.style.display = 'flex';
    loader.innerHTML = `<div style="color:var(--red,#ff4444);font-family:monospace;font-size:14px;text-align:center;padding:20px;max-width:80%;line-height:1.6;">CONNECTION FAILED<br><small style="font-size:11px;opacity:0.8;display:block;margin:10px 0;">${message}</small><br><button onclick="location.reload()" style="background:transparent;border:1px solid currentColor;color:inherit;padding:6px 16px;cursor:pointer;font-family:inherit;margin-top:10px;">RETRY</button></div>`;
}

async function ensureChannelAssignment() {
    const needsJoin = sessionStorage.getItem('cognirad_needs_channel_join') === '1';
    const storedChannel = sessionStorage.getItem('cognirad_channel');
    if (!token || (!needsJoin && storedChannel)) return;

    const joinData = await fetchJsonWithTimeout(
        `${API_BASE}/channel/join?token=${encodeURIComponent(token)}`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' } },
        8000
    );
    sessionStorage.setItem('cognirad_channel', joinData.channel_key || '');
    sessionStorage.setItem('cognirad_freq', joinData.frequency || '');
    sessionStorage.setItem('cognirad_status', joinData.status || 'FREE');
    sessionStorage.setItem('cognirad_needs_channel_join', '0');
}

// ── Helpers ───────────────────────────────────────────────────────────────
function relationForStudent(s) {
    if (!s || !s.channel_key || !myChannelKey) return 'offline';
    return s.channel_key === myChannelKey ? 'same-channel' : 'cross-channel';
}

function statusClassFor(s) {
    if (!s.channel_key) return 'offline';
    const ch = channelsState[s.channel_key];
    if (ch && ch.status === 'JAMMED') return 'jammed';
    if (s.is_online) return 'online';
    return 'away'; // has channel, no live WS
}

// ── WEB AUDIO BEEP ────────────────────────────────────────────────────────
function playNotifSound() {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.3);
    } catch (_) { }
}

// ── FORCED JAM OVERLAY ────────────────────────────────────────────────────
// Shown when admin force-jams the user's channel.
// Dismissed automatically when the user is reallocated (REALLOCATED event).

let _jamBeepInterval = null;

function showJamOverlay(channelKey) {
    const overlay = document.getElementById('jam-overlay');
    const label = document.getElementById('jam-channel-label');
    if (!overlay) return;

    isJamLocked = true;
    if (label) label.textContent = channelKey || 'CH-?';
    overlay.classList.add('active');
    if (chatInput) chatInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    // Alarm beep for 5 seconds then auto-stop (spec: "beep for 5 seconds")
    _stopJamBeep();
    _playJamBeep(); // fire immediately so there's no initial delay
    _jamBeepInterval = setInterval(() => _playJamBeep(), 800);
    // Auto-stop after 5 seconds — overlay stays visible, beep stops
    setTimeout(() => _stopJamBeep(), 5000);
}

function hideJamOverlay() {
    const overlay = document.getElementById('jam-overlay');
    isJamLocked = false;
    if (overlay) overlay.classList.remove('active');
    if (chatInput) chatInput.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    _stopJamBeep();
}

function _stopJamBeep() {
    if (_jamBeepInterval !== null) {
        clearInterval(_jamBeepInterval);
        _jamBeepInterval = null;
    }
}

function _playJamBeep() {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const gain = ctx.createGain();
        gain.connect(ctx.destination);
        gain.gain.setValueAtTime(0.0, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.35, ctx.currentTime + 0.02);
        gain.gain.setValueAtTime(0.35, ctx.currentTime + 0.28);
        gain.gain.linearRampToValueAtTime(0.0, ctx.currentTime + 0.35);

        // High tone
        const osc1 = ctx.createOscillator();
        osc1.type = 'sawtooth';
        osc1.frequency.setValueAtTime(880, ctx.currentTime);
        osc1.connect(gain);
        osc1.start(ctx.currentTime);
        osc1.stop(ctx.currentTime + 0.18);

        // Low tone follows
        const osc2 = ctx.createOscillator();
        osc2.type = 'sawtooth';
        osc2.frequency.setValueAtTime(440, ctx.currentTime + 0.18);
        osc2.connect(gain);
        osc2.start(ctx.currentTime + 0.18);
        osc2.stop(ctx.currentTime + 0.36);
    } catch (_) { }
}

// ── INITIALIZATION ────────────────────────────────────────────────────────
async function initApp() {
    if (!token || !myCms) {
        window.location.replace('/static/index.html');
        return;
    }

    document.getElementById('header-name').textContent = myName || myCms;
    document.getElementById('header-sub').textContent = `CMS: ${myCms}`;

    try {
        // Best-effort channel assignment retry to recover from transient
        // join failures during the login transition.
        await ensureChannelAssignment();

        const [studentsData, channelsData] = await Promise.all([
            fetchJsonWithTimeout(`${API_BASE}/students?token=${token}`),
            fetchJsonWithTimeout(`${API_BASE}/channel/state`)
        ]);

        allStudents = (studentsData.students || []).filter(s => s.cms !== myCms);
        channelsState = channelsData.channels || {};

        const me = (studentsData.students || []).find(s => s.cms === myCms);
        if (me && me.channel_key) {
            myChannelKey = me.channel_key;
            const myCh = channelsState[me.channel_key];
            if (myCh) {
                document.getElementById('header-freq').textContent = myCh.frequency;
                document.getElementById('header-energy').style.width = `${Math.min(100, (myCh.total_energy || 0) * 10)}%`;
                document.getElementById('header-status').textContent = myCh.status;
            }
        }

        renderContacts();
        connectWebSocket();

        // Successful initialization: always clear loader.
        finishBoot();

    } catch (e) {
        console.error('Init error', e);
        failBoot(e && e.message ? e.message : 'Unknown startup error.');
    }
}

// ── WEBSOCKET ─────────────────────────────────────────────────────────────
let _wsFailCount = 0;

function connectWebSocket() {
    ws = new WebSocket(`${WS_BASE}/ws/${token}`);
    ws.onopen = () => {
        console.log('WS Connected');
        _wsFailCount = 0;  // reset on successful connect
    };
    ws.onmessage = (e) => handleIncomingMessage(JSON.parse(e.data));
    ws.onclose = (event) => {
        // 4003 = explicit "Invalid token" from our WS handler
        // 1008 = policy violation (some proxies send this for 403)
        if (event.code === 4003 || event.code === 1008) {
            console.warn('WS auth rejected — clearing session and redirecting to login.');
            sessionStorage.clear();
            window.location.replace('/static/index.html');
            return;
        }

        // 1006 = abnormal closure (no close frame received).
        // This is what the browser sees when the server returns HTTP 403
        // during the WebSocket upgrade handshake (invalid/expired token
        // after a server restart). After 3 consecutive failures without
        // ever successfully opening, treat it as an auth failure.
        if (event.code === 1006) {
            _wsFailCount++;
            if (_wsFailCount >= 3) {
                console.warn(`WS failed ${_wsFailCount} times (code 1006) — token likely expired. Redirecting to login.`);
                sessionStorage.clear();
                window.location.replace('/static/index.html');
                return;
            }
        }

        console.log(`WS closed (code ${event.code}). Reconnecting in 3s…`);
        setTimeout(connectWebSocket, 3000);
    };
}

// ── MESSAGE STORE HELPERS ─────────────────────────────────────────────────
// All messages for a conversation are stored under the OTHER person's CMS.
// This is the single source of truth regardless of who sent the message.

function storeMessage(otherCms, msgObj) {
    if (!messages[otherCms]) messages[otherCms] = [];
    messages[otherCms].push(msgObj);
    lastActivity[otherCms] = Date.now();
}

// Append a single message bubble to the chat DOM without full re-render.
// This is the fix for the "messages not visible" bug — we never wipe the
// chat DOM for a normal message; we only append.
function appendMessageBubble(msgObj) {
    // Remove the empty-state placeholder if present
    const empty = chatMessages.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');

    if (msgObj.type === 'SYSTEM') {
        div.className = 'msg system';
        div.textContent = `⚠ ${msgObj.text}`;
    } else {
        div.className = `msg ${msgObj.isMe ? 'out' : 'in'}`;
        const timeStr = msgObj.timestamp
            ? new Date(msgObj.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const routeLabel = msgObj.route_type
            ? `<span class="msg-route">${msgObj.route_type.toUpperCase()}</span>`
            : '';
        div.innerHTML = `${routeLabel}<span class="msg-text">${msgObj.text}</span><div class="msg-time">${timeStr}</div>`;
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── INCOMING MESSAGE HANDLER ──────────────────────────────────────────────
function handleIncomingMessage(data) {

    // ── CONNECTED ──────────────────────────────────────────────────────
    if (data.type === 'CONNECTED') {
        if (data.channel_key) myChannelKey = data.channel_key;
        renderContacts();
        return;
    }

    // ── DM received (I am the recipient) ──────────────────────────────
    // Key the message under the SENDER's CMS so it appears in my chat
    // with that person.
    if (data.type === 'DM') {
        const senderCms = data.from;
        const msgObj = {
            type: 'DM',
            text: data.text,
            from: senderCms,
            to: myCms,
            isMe: false,
            timestamp: data.timestamp || new Date().toISOString(),
            route_type: data.route_type,
        };

        storeMessage(senderCms, msgObj);

        if (activeChatCms === senderCms) {
            // Chat with this person is open — append directly, no re-render
            appendMessageBubble(msgObj);
            // Mark as read immediately
            unreadCounts[senderCms] = 0;
            // Show a subtle in-chat flash (does NOT wipe the chat)
            showInChatFlash(data.from_name || senderCms, data.text);
        } else {
            // Chat is not open — show banner notification and increment badge
            unreadCounts[senderCms] = (unreadCounts[senderCms] || 0) + 1;
            showNotification(data.from_name || senderCms, data.text);
            updateBadge(senderCms, unreadCounts[senderCms]);
            renderContacts(); // re-sort so this person bubbles to top
        }
        return;
    }

    // ── MESSAGE_RESULT (confirmation that MY sent message was accepted) ─
    // Key the message under the RECIPIENT's CMS so it appears in my chat
    // with that person.
    if (data.type === 'MESSAGE_RESULT') {
        const recipientCms = data.to;

        if (data.accepted) {
            const msgObj = {
                type: 'DM',
                text: data.text,
                from: myCms,
                to: recipientCms,
                isMe: true,
                timestamp: data.timestamp || new Date().toISOString(),
                route_type: data.route_type,
            };

            storeMessage(recipientCms, msgObj);

            if (activeChatCms === recipientCms) {
                // Append directly — the message was already shown optimistically
                // by sendMessage(), so we just update the last bubble's status
                // instead of duplicating. Find the last optimistic bubble.
                const optimistic = chatMessages.querySelector('.msg.out.optimistic');
                if (optimistic) {
                    optimistic.classList.remove('optimistic');
                    optimistic.classList.add('confirmed');
                    // Update timestamp from server
                    const timeEl = optimistic.querySelector('.msg-time');
                    if (timeEl && data.timestamp) {
                        timeEl.textContent = new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    }
                }
                // If no optimistic bubble (e.g. REST path), append now
                else {
                    appendMessageBubble(msgObj);
                }
            }

            if (data.warning) showNotification('SYS_ALERT', data.warning);

        } else {
            // Message rejected — show error in chat
            const errObj = { type: 'SYSTEM', text: `REJECTED: ${data.warning || 'Channel overloaded'}` };
            storeMessage(recipientCms, errObj);

            if (activeChatCms === recipientCms) {
                // Remove the optimistic bubble since it was rejected
                const optimistic = chatMessages.querySelector('.msg.out.optimistic');
                if (optimistic) optimistic.remove();
                appendMessageBubble(errObj);
            }
        }
        return;
    }

    // ── REALLOCATED ────────────────────────────────────────────────────
    if (data.type === 'REALLOCATED') {
        myChannelKey = data.to || myChannelKey;
        if (data.frequency) document.getElementById('header-freq').textContent = data.frequency;
        // Dismiss the jam overlay — user has been moved to a new channel
        hideJamOverlay();
        renderContacts();
        const sysMsg = { type: 'SYSTEM', text: `REALLOCATED → ${data.to}  FREQ: ${data.frequency}` };
        if (activeChatCms) {
            storeMessage(activeChatCms, sysMsg);
            appendMessageBubble(sysMsg);
        }
        showNotification('REALLOCATED', `Moved to ${data.to} — ${data.frequency}`);
        return;
    }

    // ── ERROR ──────────────────────────────────────────────────────────
    if (data.type === 'ERROR') {
        showNotification('SYS_ALERT', data.detail || 'Transmission rejected.');
        if (activeChatCms) {
            const errObj = { type: 'SYSTEM', text: data.detail || 'Transmission rejected.' };
            storeMessage(activeChatCms, errObj);
            appendMessageBubble(errObj);
        }
        return;
    }

    // ── SYSTEM ────────────────────────────────────────────────────────
    if (data.type === 'SYSTEM') {
        // Force-jam from admin → show full-screen jam overlay
        if (data.subtype === 'CHANNEL_JAMMED') {
            showJamOverlay(data.channel_key || myChannelKey);
            return;
        }
        if (activeChatCms) {
            const sysObj = { type: 'SYSTEM', text: `SYSTEM: ${data.subtype}` };
            storeMessage(activeChatCms, sysObj);
            appendMessageBubble(sysObj);
        }
    }
}

// ── CONTACTS RENDER ───────────────────────────────────────────────────────
function renderContacts() {
    contactsList.innerHTML = '';

    // 1. Apply search filter
    const q = searchQuery.toLowerCase().trim();
    let visible = q
        ? allStudents.filter(s => (s.name || s.cms).toLowerCase().includes(q))
        : [...allStudents];

    // 2. Apply tab filter
    if (activeFilter === 'same') {
        visible = visible.filter(s => relationForStudent(s) === 'same-channel');
    } else if (activeFilter === 'other') {
        visible = visible.filter(s => relationForStudent(s) === 'cross-channel');
    } else if (activeFilter === 'online') {
        visible = visible.filter(s => s.is_online || s.channel_key);
    }

    if (!visible.length) {
        contactsList.innerHTML = `<div class="empty-state" style="padding:24px;">${q ? `No results for "${q}"` : 'No students match this filter.'}</div>`;
        return;
    }

    // 3. Sort: people with recent messages first (by lastActivity desc),
    //    then alphabetically within each group.
    visible.sort((a, b) => {
        const ta = lastActivity[a.cms] || 0;
        const tb = lastActivity[b.cms] || 0;
        if (tb !== ta) return tb - ta; // most recent first
        return (a.name || a.cms).localeCompare(b.name || b.cms);
    });

    // 4. When filter is 'all', group into sections
    if (activeFilter === 'all') {
        const sameChannel = visible.filter(s => relationForStudent(s) === 'same-channel');
        const otherChannel = visible.filter(s => relationForStudent(s) === 'cross-channel');
        const offline = visible.filter(s => relationForStudent(s) === 'offline');

        buildSection(`SAME CHANNEL  (${sameChannel.length})`, sameChannel, 'section-same');
        buildSection(`OTHER CHANNELS  (${otherChannel.length})`, otherChannel, 'section-other');
        buildSection(`OFFLINE  (${offline.length})`, offline, 'section-offline');
    } else {
        // Flat list for filtered views
        visible.forEach(s => contactsList.appendChild(buildRow(s)));
    }
}

function buildRow(s) {
    const sc = statusClassFor(s);
    const relation = relationForStudent(s);
    const ch = s.channel_key ? channelsState[s.channel_key] : null;
    const chLabel = s.channel_key || null;
    const freq = ch ? ch.frequency : null;
    const unread = unreadCounts[s.cms] || 0;

    const row = document.createElement('div');
    row.className = `contact-row route-${relation}`;

    const chTag = chLabel
        ? `<span class="ch-tag">${chLabel}</span>`
        : `<span class="ch-tag offline-tag">OFFLINE</span>`;

    const freqLine = freq
        ? `<span class="c-freq">${freq}</span>`
        : `<span class="c-freq muted">—</span>`;

    // Last message preview
    const convMsgs = messages[s.cms] || [];
    const lastMsg = convMsgs.length ? convMsgs[convMsgs.length - 1] : null;
    const preview = lastMsg && lastMsg.type !== 'SYSTEM'
        ? `<span class="c-preview">${lastMsg.isMe ? 'You: ' : ''}${lastMsg.text.slice(0, 28)}${lastMsg.text.length > 28 ? '…' : ''}</span>`
        : '';

    row.innerHTML = `
        <div class="status-indicator ${sc}"></div>
        <div class="c-info">
            <div class="c-name-row">
                <span class="c-name">${s.name || s.cms}</span>
                ${chTag}
            </div>
            <div class="c-sub-row">
                ${preview || freqLine}
            </div>
        </div>
        <div class="c-right">
            <div class="c-badge${unread > 0 ? ' active' : ''}" id="badge-${s.cms}">${unread > 0 ? unread : ''}</div>
        </div>
    `;
    row.onclick = () => openChat(s);
    return row;
}

function buildSection(title, students, cls) {
    if (!students.length) return;
    const header = document.createElement('div');
    header.className = `section-header ${cls}`;
    header.textContent = title;
    contactsList.appendChild(header);
    students.forEach(s => contactsList.appendChild(buildRow(s)));
}

// ── FILTER TABS ───────────────────────────────────────────────────────────
function setFilter(f) {
    activeFilter = f;
    document.querySelectorAll('.filter-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === f);
    });
    renderContacts();
}

// ── SEARCH ────────────────────────────────────────────────────────────────
if (searchInput) {
    searchInput.addEventListener('input', () => {
        searchQuery = searchInput.value;
        if (searchClear) searchClear.classList.toggle('visible', searchQuery.length > 0);
        renderContacts();
    });
}

if (searchClear) {
    searchClear.addEventListener('click', () => {
        searchInput.value = '';
        searchQuery = '';
        searchClear.classList.remove('visible');
        searchInput.focus();
        renderContacts();
    });
}

// ── CHAT OPEN / CLOSE ─────────────────────────────────────────────────────
function openChat(student) {
    activeChatCms = student.cms;

    document.getElementById('chat-target-name').textContent = student.name || student.cms;
    const relation = relationForStudent(student);
    const ch = student.channel_key ? channelsState[student.channel_key] : null;
    const relLabel = relation === 'same-channel' ? 'SAME CHANNEL'
        : relation === 'cross-channel' ? 'CROSS CHANNEL' : 'OFFLINE';
    document.getElementById('chat-target-sub').textContent = student.channel_key
        ? `${relLabel}  ·  ${student.channel_key}  ·  ${(ch && ch.frequency) ? ch.frequency : ''}`
        : 'OFFLINE';

    // Clear unread
    unreadCounts[student.cms] = 0;
    const badge = document.getElementById(`badge-${student.cms}`);
    if (badge) { badge.textContent = ''; badge.classList.remove('active'); }

    document.body.classList.add('viewing-chat');

    // Full render of existing messages (only on open, not on every new message)
    renderMessages(student.cms);
}

backBtn.onclick = () => {
    activeChatCms = null;
    document.body.classList.remove('viewing-chat');
    renderContacts();
};

// Full re-render of the chat area — only called when opening a chat.
// For live incoming/outgoing messages we use appendMessageBubble() instead.
function renderMessages(cms) {
    chatMessages.innerHTML = '';
    const msgs = messages[cms] || [];

    if (!msgs.length) {
        chatMessages.innerHTML = `<div class="empty-state">Secure channel established. Awaiting transmission...</div>`;
        return;
    }

    msgs.forEach(m => {
        const div = document.createElement('div');
        if (m.type === 'SYSTEM') {
            div.className = 'msg system';
            div.textContent = `⚠ ${m.text}`;
        } else {
            div.className = `msg ${m.isMe ? 'out' : 'in'}`;
            const timeStr = m.timestamp
                ? new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                : '';
            const routeLabel = m.route_type
                ? `<span class="msg-route">${m.route_type.toUpperCase()}</span>`
                : '';
            div.innerHTML = `${routeLabel}<span class="msg-text">${m.text}</span><div class="msg-time">${timeStr}</div>`;
        }
        chatMessages.appendChild(div);
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── SENDING ───────────────────────────────────────────────────────────────
function sendMessage() {
    const text = chatInput.value.trim();
    if (isJamLocked || !text || !activeChatCms || !ws || ws.readyState !== WebSocket.OPEN) return;

    // Show message optimistically immediately so the sender sees it right away.
    // It will be confirmed (or removed) when MESSAGE_RESULT arrives.
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const empty = chatMessages.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = 'msg out optimistic';
    div.innerHTML = `<span class="msg-text">${text}</span><div class="msg-time">${timeStr} <span class="msg-sending">⏳</span></div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    ws.send(JSON.stringify({ to: activeChatCms, text }));
    chatInput.value = '';
    chatInput.focus();
}

sendBtn.onclick = sendMessage;
chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

// ── NOTIFICATIONS ─────────────────────────────────────────────────────────
let notifTimeout;

function showNotification(title, text) {
    notifTitle.textContent = title;
    notifBody.textContent = text;
    notifBanner.classList.add('show');
    playNotifSound();
    if (navigator.vibrate) navigator.vibrate([100, 50, 100]);
    clearTimeout(notifTimeout);
    notifTimeout = setTimeout(() => notifBanner.classList.remove('show'), 4000);
}

// Subtle flash at the top of the chat — does NOT wipe or re-render messages.
// Only shown when the chat with the sender is already open.
function showInChatFlash(senderName, text) {
    const flash = document.createElement('div');
    flash.className = 'in-chat-notif';
    flash.innerHTML = `<strong>${senderName}</strong>: ${text.length > 40 ? text.slice(0, 40) + '…' : text}`;
    // Insert at top of chat scroll, not prepend to chatMessages
    chatMessages.insertBefore(flash, chatMessages.firstChild);
    setTimeout(() => { if (flash.parentNode) flash.remove(); }, 3000);
}

notifBanner.addEventListener('click', () => {
    notifBanner.classList.remove('show');
    clearTimeout(notifTimeout);
});

function updateBadge(cms, count) {
    const badge = document.getElementById(`badge-${cms}`);
    if (badge) {
        badge.textContent = count > 0 ? count : '';
        badge.classList.toggle('active', count > 0);
    }
}

// ── MOBILE VIEWPORT FIX ───────────────────────────────────────────────────
if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', () => {
        document.body.style.height = `${window.visualViewport.height}px`;
        window.scrollTo(0, 0);
    });
}

// Global guards so unexpected runtime errors never leave the loader hanging.
window.addEventListener('error', (event) => {
    console.error('Fatal error during app boot:', event.error || event.message);
    if (!bootCompleted) {
        failBoot((event.error && event.error.message) || event.message || 'Unexpected JavaScript error.');
    }
});

window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    const message = (reason && reason.message) ? reason.message : String(reason || 'Unhandled promise rejection.');
    console.error('Unhandled rejection during app boot:', reason);
    if (!bootCompleted) {
        failBoot(message);
    }
});

// Hard watchdog: never allow an infinite "ESTABLISHING CONNECTION..." state.
bootWatchdogId = setTimeout(() => {
    if (!bootCompleted) {
        failBoot('Startup timed out. Backend reachable but app initialization did not complete.');
    }
}, 15000);

initApp();

// ── BOT SIMULATION MODE ───────────────────────────────────────────────────
if (sessionStorage.getItem('cognirad_bot') === 'true') {
    console.log('[BOT MODE] Simulation active. Will send automated messages.');

    const CHITCHAT = [
        "Hey, how is your project going?",
        "Did you see the new assignment?",
        "I'm testing the cognitive radio system.",
        "Can anyone help me with allocator.py?",
        "Let's meet at the library later.",
        "Is the Wi-Fi slow for you guys too?",
        "Sending a small ping to check routing.",
        "All good here.",
        "Got it, thanks!"
    ];
    const HEAVY_PAYLOAD = "This is a massive payload designed to simulate heavy file transfer over the network. ".repeat(10);

    // Baseline Chatter
    setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN || !allStudents.length) return;
        // Don't send if we happen to be viewing someone's chat (so the UI doesn't get flooded while we watch)
        // Wait, it's actually fun to see it happen in real time, but let's just let it send randomly!

        // Pick a random student (prefer online ones)
        const online = allStudents.filter(s => s.is_online || s.channel_key);
        const pool = online.length > 0 ? online : allStudents;
        const target = pool[Math.floor(Math.random() * pool.length)];

        const msg = CHITCHAT[Math.floor(Math.random() * CHITCHAT.length)];
        ws.send(JSON.stringify({ to: target.cms, text: msg }));

    }, 2000 + Math.random() * 3000); // Every 2-5 seconds

    // Congestion Bursts (Rarely send a massive payload)
    setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN || !allStudents.length) return;
        // 20% chance to burst every 15 seconds
        if (Math.random() < 0.2) {
            console.log('[BOT MODE] Firing congestion burst!');
            const target = allStudents[Math.floor(Math.random() * allStudents.length)];

            // Fire 5 massive payloads in rapid succession
            for (let i = 0; i < 5; i++) {
                setTimeout(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ to: target.cms, text: HEAVY_PAYLOAD }));
                    }
                }, i * 100);
            }
        }
    }, 15000);
}
