// static/js/app.js

const API_BASE = window.location.origin;
const WS_BASE = API_BASE.replace(/^http/, 'ws');
const token = sessionStorage.getItem('cognirad_token');
const myCms = sessionStorage.getItem('cognirad_cms');
const myName = sessionStorage.getItem('cognirad_name');

// DOM Elements
const loader = document.getElementById('app-loader');
const contactsList = document.getElementById('contacts-list');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('chat-send-btn');
const backBtn = document.getElementById('chat-back-btn');
const notifBanner = document.getElementById('notif-banner');
const notifTitle = document.getElementById('notif-title');
const notifBody = document.getElementById('notif-body');

// State
let allStudents = [];
let channelsState = {};
let activeChatCms = null;
let messages = {}; // cms -> [msg objects]
let ws = null;

// --- INITIALIZATION ---
async function initApp() {
    // Redirect to login if no token or CMS
    if (!token || !myCms) {
        window.location.replace('/static/index.html');
        return;
    }

    // Set Header Info
    document.getElementById('header-name').textContent = myName || myCms;
    document.getElementById('header-sub').textContent = `CMS: ${myCms}`;
    
    try {
        // Fetch data
        const [studentsRes, channelsRes] = await Promise.all([
            fetch(`${API_BASE}/admin/students`),
            fetch(`${API_BASE}/channel/state`)
        ]);

        if (!studentsRes.ok) throw new Error(`Students fetch failed: ${studentsRes.status}`);
        if (!channelsRes.ok) throw new Error(`Channels fetch failed: ${channelsRes.status}`);
        
        const studentsData = await studentsRes.json();
        const channelsData = await channelsRes.json();
        
        allStudents = (studentsData.students || []).filter(s => s.cms !== myCms);
        channelsState = channelsData.channels || {};
        
        // Update my channel info in header
        const me = (studentsData.students || []).find(s => s.cms === myCms);
        if (me && me.channel_key) {
            const myCh = channelsState[me.channel_key];
            if (myCh) {
                document.getElementById('header-freq').textContent = myCh.frequency;
                document.getElementById('header-energy').style.width = `${Math.min(100, (myCh.total_energy || 0) * 10)}%`;
                document.getElementById('header-status').textContent = myCh.status;
            }
        }
        
        renderContacts();
        connectWebSocket();
        
        // Hide loader with transition
        loader.style.opacity = '0';
        loader.addEventListener('transitionend', () => {
            loader.style.display = 'none';
        }, { once: true });
        
    } catch (e) {
        console.error("Init error", e);
        // Keep loader visible but show error message
        loader.style.opacity = '1';
        loader.style.display = 'flex';
        loader.innerHTML = `<div style="color:var(--red,#ff4444); font-family: monospace; font-size: 14px; text-align:center; padding: 20px; max-width: 80%; line-height: 1.6;">CONNECTION FAILED<br><small style="font-size:11px;opacity:0.7;display:block;margin:10px 0;">${e.message}</small><br><button onclick="location.reload()" style="background:transparent;border:1px solid currentColor;color:inherit;padding:6px 16px;cursor:pointer;font-family:inherit;margin-top:10px;">RETRY</button></div>`;
    }
}

// --- WEBSOCKET ---
function connectWebSocket() {
    ws = new WebSocket(`${WS_BASE}/ws/${token}`);
    
    ws.onopen = () => console.log("WS Connected");
    
    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        handleIncomingMessage(data);
    };
    
    ws.onclose = () => {
        console.log("WS Closed. Reconnecting in 3s...");
        setTimeout(connectWebSocket, 3000);
    };
}

function handleIncomingMessage(data) {
    if (data.type === 'DM') {
        const sender = data.from;
        if (!messages[sender]) messages[sender] = [];
        messages[sender].push(data);
        
        if (activeChatCms === sender) {
            renderMessages(sender);
        } else {
            showNotification(data.from_name || sender, data.text);
            updateBadge(sender, 1);
        }
    } else if (data.type === 'MESSAGE_RESULT') {
        if (data.accepted) {
            if (!messages[data.to]) messages[data.to] = [];
            messages[data.to].push({ type: 'DM', text: data.text, from: myCms, isMe: true, timestamp: data.timestamp });
            if (activeChatCms === data.to) renderMessages(data.to);
        } else {
            // Show system error in chat
            if (!messages[data.to]) messages[data.to] = [];
            messages[data.to].push({ type: 'SYSTEM', text: `ERROR: ${data.warning}` });
            if (activeChatCms === data.to) renderMessages(data.to);
        }
    } else if (data.type === 'REALLOCATED') {
        // I got moved
        document.getElementById('header-freq').textContent = data.frequency;
        appendSystemMessage(activeChatCms || "SYSTEM", `SYSTEM REALLOCATED YOU TO ${data.frequency}`);
    } else if (data.type === 'SYSTEM') {
        // e.g. someone jammed or joined
        if (activeChatCms) appendSystemMessage(activeChatCms, `SYSTEM: ${data.subtype}`);
    }
}

function appendSystemMessage(cms, text) {
    if (!messages[cms]) messages[cms] = [];
    messages[cms].push({ type: 'SYSTEM', text: text });
    if (activeChatCms === cms) renderMessages(cms);
}

// --- UI LOGIC ---
function renderContacts() {
    contactsList.innerHTML = '';
    allStudents.forEach(s => {
        const row = document.createElement('div');
        row.className = 'contact-row';
        
        // Status indicator
        let statusClass = 'offline';
        if (s.channel_key) {
            const ch = channelsState[s.channel_key];
            if (ch && ch.status === 'JAMMED') statusClass = 'jammed';
            else statusClass = 'online';
        }
        
        row.innerHTML = `
            <div class="status-indicator ${statusClass}"></div>
            <div class="c-info">
                <div class="c-name">${s.name || s.cms}</div>
                <div class="c-sub">FREQ: ${s.channel_key ? channelsState[s.channel_key]?.frequency || 'UNKNOWN' : 'OFFLINE'}</div>
            </div>
            <div class="c-badge" id="badge-${s.cms}">0</div>
        `;
        row.onclick = () => openChat(s);
        contactsList.appendChild(row);
    });
}

function openChat(student) {
    activeChatCms = student.cms;
    document.getElementById('chat-target-name').textContent = student.name || student.cms;
    document.getElementById('chat-target-sub').textContent = student.channel_key ? `FREQ: ${channelsState[student.channel_key]?.frequency}` : 'OFFLINE';
    
    // Clear badge
    const badge = document.getElementById(`badge-${student.cms}`);
    if (badge) {
        badge.textContent = '0';
        badge.classList.remove('active');
    }
    
    document.body.classList.add('viewing-chat');
    renderMessages(student.cms);
}

backBtn.onclick = () => {
    activeChatCms = null;
    document.body.classList.remove('viewing-chat');
};

function renderMessages(cms) {
    chatMessages.innerHTML = '';
    const msgs = messages[cms] || [];
    if (msgs.length === 0) {
        chatMessages.innerHTML = `<div class="empty-state">Secure channel established. Awaiting transmission...</div>`;
        return;
    }
    
    msgs.forEach(m => {
        const div = document.createElement('div');
        if (m.type === 'SYSTEM') {
            div.className = 'msg system';
            div.textContent = `[SYS_ALERT] ${m.text}`;
        } else {
            div.className = `msg ${m.isMe ? 'out' : 'in'}`;
            const timeStr = m.timestamp ? new Date(m.timestamp).toLocaleTimeString() : '';
            div.innerHTML = `<div class="msg-time">${timeStr}</div>${m.text}`;
        }
        chatMessages.appendChild(div);
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- SENDING ---
function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || !activeChatCms || !ws) return;
    
    ws.send(JSON.stringify({
        to: activeChatCms,
        text: text
    }));
    chatInput.value = '';
    chatInput.focus();
}

sendBtn.onclick = sendMessage;
chatInput.onkeypress = (e) => { if (e.key === 'Enter') sendMessage(); };

// --- NOTIFICATIONS ---
let notifTimeout;
function showNotification(title, text) {
    notifTitle.textContent = title;
    notifBody.textContent = text;
    notifBanner.classList.add('show');
    
    // Vibrate
    if (navigator.vibrate) navigator.vibrate(200);
    
    // Audio (requires an asset to be physically present, using a fallback beep if missing is harder in pure JS without context, so we just attempt it silently)
    try {
        new Audio('/static/assets/notification.mp3').play().catch(() => {});
    } catch(e) {}
    
    clearTimeout(notifTimeout);
    notifTimeout = setTimeout(() => {
        notifBanner.classList.remove('show');
    }, 3000);
}

function updateBadge(cms, addCount) {
    const badge = document.getElementById(`badge-${cms}`);
    if (badge) {
        const current = parseInt(badge.textContent) || 0;
        badge.textContent = current + addCount;
        badge.classList.add('active');
    }
}

// Fix input scroll on mobile
if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', () => {
        document.body.style.height = `${window.visualViewport.height}px`;
        window.scrollTo(0, 0);
    });
}

// Start
initApp();
