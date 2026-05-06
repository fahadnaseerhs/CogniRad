# CogniRad — Manual Testing Checklist

This document is a step-by-step guide for verifying every feature of the CogniRad system before production deployment.

---

## Pre-flight checks

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `students.json` exists in project root with at least 10 CMS IDs
- [ ] Server running at `http://localhost:8080` (command: `python main.py`)
- [ ] Browser DevTools open (F12) for debugging

---

## 1. Student login and channel assignment

**Goal:** Verify students can log in and are assigned to a channel.

### Steps
1. Open `http://localhost:8080/static/index.html`
2. Enter a valid CMS ID from `students.json` (e.g. `100001`)
3. Click **Login**

### Expected results
- [ ] Redirect to `app.html` within 1 second
- [ ] Header shows student name (e.g. "Alice Johnson")
- [ ] Header shows channel assignment (e.g. "CH-1")
- [ ] Header shows frequency (e.g. "2.412 GHz")
- [ ] Energy bar in header is at 0% (empty)
- [ ] Contacts list loads all other students
- [ ] Online students show a green status dot

### Failure indicators
- **"Invalid CMS"** → CMS not in `students.json`
- **Redirect loop** → Token not stored in sessionStorage (check DevTools → Application → Session Storage)
- **Blank screen** → JavaScript error (check DevTools → Console)

---

## 2. Direct messaging (same channel)

**Goal:** Verify students on the same channel can send DMs.

### Steps
1. Log in as Student A in Tab 1
2. Log in as Student B in Tab 2 (should be on same channel as A)
3. In Tab 1, click Student B in the contacts list
4. Type a message and click Send
5. Observe both tabs

### Expected results
- [ ] Message appears in Tab 1 chat with green left border (DELIVERED)
- [ ] Message appears in Tab 2 chat instantly
- [ ] Tab 1 header energy bar increases slightly
- [ ] Tab 2 shows an unread badge on Student A's contact card
- [ ] Terminal dashboard shows the message in the live feed

### Failure indicators
- **"You are not assigned to a channel"** → Student never joined (call `POST /channel/join`)
- **Message shows REJECTED** → Channel is JAMMED
- **Message never appears in Tab 2** → WebSocket disconnected (check DevTools → Network → WS)

---

## 3. Direct messaging (cross-channel)

**Goal:** Verify students on different channels can send DMs.

### Steps
1. Log in as Student A on CH-1
2. Log in as Student B on CH-2 (use simulation or manual reallocation to force different channels)
3. Send a message from A to B

### Expected results
- [ ] Message is delivered (cross-channel DMs are allowed)
- [ ] Message metadata shows `route_type: "cross-channel"`
- [ ] Both sender and recipient see the same channel status and SNR in signal metadata

---

## 4. Admin dashboard login

**Goal:** Verify admin can access the dashboard.

### Steps
1. Open `http://localhost:8080/admin`
2. Enter password (default: `admin`)
3. Click **Unlock**

### Expected results
- [ ] Login panel disappears
- [ ] Dashboard appears with all panels visible
- [ ] Clock in topbar ticks every second (format: `HH:MM:SS`)
- [ ] Feed badge shows `● LIVE` with green dot within 2 seconds
- [ ] All 5 channel cards render with status pills
- [ ] Metric strip shows correct counts (channels=5, students=N, online=M)

### Failure indicators
- **"Invalid admin key"** → Wrong password or `COGNIRAD_ADMIN_PASSWORD` env var set
- **Feed badge stays `POLLING`** → WebSocket `/ws/spectrum` blocked (firewall/proxy issue)
- **Charts show "Waiting for data…" after 5+ seconds** → WebSocket not connected

---

## 5. Live channel energy graphs

**Goal:** Verify charts update in real time as students send messages.

### Steps
1. Log in as admin
2. Open a second browser tab, log in as a student
3. Send 5–10 messages rapidly from the student tab
4. Watch the admin dashboard

### Expected results
- [ ] **Energy per Channel** chart: student's channel line rises with each message
- [ ] **SNR** chart: SNR drops as energy rises on that channel
- [ ] **Predictive AI — Energy Slope** chart: slope goes positive during rapid messaging
- [ ] **Round-Robin Allocator** chart: occupancy line stays flat (no reallocation yet)
- [ ] All charts update every 1 second
- [ ] Chart legends show all 5 channels with correct colors

### Failure indicators
- **Charts frozen** → WebSocket disconnected (check console for errors)
- **Charts show NaN or Infinity** → Data format mismatch (check `SPECTRUM_TICK` payload)

---

## 6. Forced jam (admin → student)

**Goal:** Verify admin can force-jam a channel and students are blocked + reallocated.

### Steps
1. Have at least one student logged in on CH-1
2. In admin dashboard, click **Jam** on CH-1
3. Observe both the admin dashboard and the student's browser

### Expected results — Admin dashboard
- [ ] CH-1 status pill changes to `JAMMED` (purple)
- [ ] CH-1 card border pulses purple
- [ ] Jam button shows `⚡ JAMMED` and pulses
- [ ] Event log shows: `⚡ CH-1 FORCE JAMMED — N student(s) evacuated`
- [ ] Live feed shows a `CHANNEL JAMMED` row
- [ ] Students on CH-1 appear on a different channel in the students table after 1–3 seconds

### Expected results — Student browser
- [ ] Full-screen overlay appears immediately: "CHANNEL JAMMED / CH-1 / SPECTRUM INTERFERENCE DETECTED"
- [ ] Two-tone alarm beep plays for 5 seconds then stops automatically
- [ ] Chat input is blocked (overlay covers entire screen)
- [ ] After 1–3 seconds, overlay disappears automatically when student is reallocated
- [ ] Header updates to show new channel and frequency
- [ ] System message appears in chat: `REALLOCATED → CH-3  FREQ: 2.462 GHz`

### Failure indicators
- **Overlay does not appear** → `SYSTEM/CHANNEL_JAMMED` event not received (check DevTools → Network → WS)
- **Beep does not play** → Browser blocked AudioContext (user must interact with page first)
- **Overlay never dismisses** → `REALLOCATED` event not received (check allocator logs)
- **Beep runs forever** → `setTimeout` not working (check `app.js` line 95)

---

## 7. Unjam (admin)

**Goal:** Verify admin can clear a jammed channel.

### Steps
1. After jamming CH-1, click **Unjam** on CH-1

### Expected results
- [ ] CH-1 status returns to `FREE`
- [ ] Jam button no longer pulses
- [ ] Event log shows: `✓ CH-1 unjammed — channel restored to FREE`
- [ ] Students can be assigned back to CH-1 on next login or reallocation

---

## 8. Manual reallocation (admin)

**Goal:** Verify admin can manually trigger reallocation.

### Steps
1. Have 3+ students on one channel
2. Click **Reallocate** on that channel

### Expected results
- [ ] At least one student moves to a different channel
- [ ] Event log shows: `↺ CH-X reallocated — N student(s) moved`
- [ ] Reallocation chip appears below the Round-Robin chart (format: `CMS001 CH-1→CH-3`)
- [ ] Moved students receive a `REALLOCATED` WebSocket event in their browser
- [ ] Students table updates to show new channel assignments

---

## 9. AI automatic reallocation

**Goal:** Verify the AI loop automatically reallocates overloaded channels.

### Steps
1. In admin dashboard, set **Energy per student** to `15` and click **Run**
2. Watch the channel cards and event log for 5–10 seconds

### Expected results
- [ ] Channels with high energy (CONGESTED or JAMMED) trigger automatic reallocation within 1–3 seconds
- [ ] Event log shows reallocation events
- [ ] Reallocation chips appear below the member chart
- [ ] `Reallocations` metric in the strip increments
- [ ] After reallocation, channel status recovers to BUSY or FREE

---

## 10. Active students energy panel

**Goal:** Verify the energy bars update as students send messages.

### Steps
1. Have several students logged in and sending messages
2. Watch the **Active Students** panel on the right side of the admin dashboard

### Expected results
- [ ] Online students show a green dot
- [ ] Offline students show a gray dot
- [ ] Energy bars fill left-to-right as students send messages
- [ ] Bar color changes: green (< 50% of max), yellow (50–80%), red (≥ 80%)
- [ ] Students sorted: online first, then by energy descending
- [ ] Values update every 2 seconds

---

## 11. Simulation load test

**Goal:** Verify the simulation endpoint seeds a realistic classroom load.

### Steps
1. In admin dashboard, set **Energy per student** to `25` and click **Run**
2. Wait 3 seconds

### Expected results
- [ ] 25 students assigned (5 per channel)
- [ ] Each channel shows 5 users and elevated energy
- [ ] Some channels may immediately show CONGESTED or JAMMED
- [ ] AI loop triggers reallocation within 1–3 seconds
- [ ] All charts update to reflect the new load
- [ ] Event log shows: `Simulation: 25.0 J per student applied`

---

## 12. Student search

**Goal:** Verify the students table search filter works.

### Steps
1. In the **Students** table, type a CMS ID or partial name in the search box

### Expected results
- [ ] Table filters in real time (no delay)
- [ ] Matching rows remain visible
- [ ] Non-matching rows disappear
- [ ] Clear the search to restore all rows

---

## 13. Session persistence (admin)

**Goal:** Verify admin session persists across page refreshes.

### Steps
1. Log in to admin dashboard
2. Refresh the page (F5)

### Expected results
- [ ] Dashboard unlocks automatically without re-entering the password
- [ ] All data reloads
- [ ] WebSocket reconnects within 2 seconds

### Steps
1. Click **Lock**
2. Refresh the page

### Expected results
- [ ] Login panel is shown
- [ ] No auto-unlock

---

## 14. Terminal dashboard (server-side)

**Goal:** Verify the ASCII dashboard renders correctly in the terminal.

### Steps
1. Watch the terminal where `python main.py` is running

### Expected results (every 4 seconds)
- [ ] ASCII CogniRad banner with server URL in red
- [ ] Channel energy bars colored by status:
  - Green = FREE
  - Yellow = BUSY
  - Red = CONGESTED
  - Magenta = JAMMED
- [ ] Active students with energy bars (green → yellow → red as energy rises)
- [ ] Live message feed showing last 8 messages with:
  - Timestamp
  - Sender → Recipient
  - Channel
  - Energy
  - Status
  - Delivery outcome (color-coded)
  - PHY telemetry (bitrate, utilization, modulation)

---

## 15. Automated test suite

**Goal:** Verify all 59 tests pass.

### Steps
```bash
pytest test_all.py --asyncio-mode=auto -v
```

### Expected results
- [ ] `59 passed` in under 5 seconds
- [ ] 0 failures
- [ ] 0 errors
- [ ] 0 warnings

### Failure indicators
- **Any test fails** → Regression introduced, check git diff
- **Warnings about `utcnow()`** → Not fixed correctly in `main.py`

---

## 16. Browser compatibility

**Goal:** Verify the system works in all major browsers.

### Browsers to test
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Edge (latest)
- [ ] Safari (macOS/iOS)

### Steps
1. Log in as a student in each browser
2. Send a message
3. Log in as admin in each browser
4. Verify charts render

### Expected results
- [ ] All features work identically in all browsers
- [ ] No console errors
- [ ] Charts render correctly (Canvas 2D is universally supported)

---

## 17. Mobile responsiveness

**Goal:** Verify the student app works on mobile devices.

### Steps
1. Open `http://localhost:8080/static/index.html` on a mobile device or use DevTools → Device Toolbar
2. Log in as a student
3. Send a message

### Expected results
- [ ] Login panel is readable and usable
- [ ] Chat interface is usable (no horizontal scroll)
- [ ] Contacts list is scrollable
- [ ] Jam overlay covers the entire screen

**Note:** The admin dashboard is **not** mobile-optimized (minimum width: 1200px). It is designed for desktop use only.

---

## 18. WebSocket reconnection

**Goal:** Verify WebSocket reconnects automatically after a disconnect.

### Steps
1. Log in as a student
2. In DevTools → Network → WS, right-click the WebSocket connection and select "Close connection"
3. Wait 5 seconds
4. Send a message

### Expected results
- [ ] WebSocket reconnects automatically (check Network tab)
- [ ] Message is delivered successfully

**Note:** The current implementation does **not** have automatic reconnection logic. This is a known limitation. If the WebSocket disconnects, the user must refresh the page.

---

## 19. Concurrent users stress test

**Goal:** Verify the system handles 25+ concurrent users.

### Steps
1. Run: `python main.py --simulate`
2. Wait for all 25 browser tabs to open and log in
3. Watch the admin dashboard and terminal dashboard

### Expected results
- [ ] All 25 students log in successfully
- [ ] Students are distributed across all 5 channels (5 per channel)
- [ ] AI loop triggers reallocations as channels become overloaded
- [ ] No server crashes or errors in the terminal
- [ ] Admin dashboard remains responsive

---

## 20. Edge cases

### 20.1 Empty channel reallocation

**Steps:**
1. Manually call `POST /admin/reallocate` with `channel_key: "CH-1"` when CH-1 has 0 users

**Expected:**
- [ ] Returns `{"channel": "CH-1", "moved": []}`
- [ ] No errors

### 20.2 All channels jammed

**Steps:**
1. Jam all 5 channels via admin dashboard
2. Try to send a message as a student

**Expected:**
- [ ] Message is rejected with `REJECTED_CHANNEL_JAMMED`
- [ ] Student is **not** reallocated (no valid destination)

### 20.3 Student logs out mid-reallocation

**Steps:**
1. Start a reallocation
2. Immediately log out the student being moved

**Expected:**
- [ ] No server crash
- [ ] Student is removed from the old channel
- [ ] Student is **not** added to the new channel (session invalidated)

---

## Production readiness checklist

- [ ] All 59 automated tests pass
- [ ] All 20 manual test sections pass
- [ ] No console errors in any browser
- [ ] No Python exceptions in the terminal
- [ ] Admin password changed from default `admin` (set `COGNIRAD_ADMIN_PASSWORD` env var)
- [ ] `students.json` populated with real student data
- [ ] Database file `cognirad.db` backed up
- [ ] Server runs on a stable port (default: 8080)
- [ ] Firewall allows WebSocket connections
- [ ] HTTPS configured if deploying to production (use a reverse proxy like nginx)

---

## Known limitations

1. **No WebSocket auto-reconnect** — If the WebSocket disconnects, the user must refresh the page.
2. **Admin dashboard not mobile-optimized** — Minimum width: 1200px. Use on desktop only.
3. **No message persistence** — Messages are stored in the database but not loaded on login (only new messages are shown).
4. **No user presence heartbeat** — If a user closes the tab without logging out, they remain "online" until the WebSocket times out (30–60 seconds).
5. **No rate limiting** — A malicious user can spam messages and overload a channel. Consider adding rate limiting in production.

---

## Troubleshooting

### WebSocket connection fails

**Symptoms:** Feed badge shows `POLLING`, charts don't update, messages don't arrive.

**Fixes:**
- Check firewall allows WebSocket connections
- Check browser console for CORS errors
- Verify server is running on the correct port
- Try a different browser

### Charts show "Waiting for data…" forever

**Symptoms:** Admin dashboard charts never populate.

**Fixes:**
- Check WebSocket `/ws/spectrum` is connected (DevTools → Network → WS)
- Check server logs for errors in `_ai_loop()`
- Verify `SPECTRUM_TICK` payload is being sent (add `print(payload)` in `_ai_loop()`)

### Jam overlay doesn't appear

**Symptoms:** Student doesn't see the jam overlay when admin jams their channel.

**Fixes:**
- Check student WebSocket is connected
- Check `SYSTEM/CHANNEL_JAMMED` event is being sent (add logging in `jam_channel()`)
- Check `showJamOverlay()` is being called (add `console.log` in `app.js`)

### Beep doesn't play

**Symptoms:** Jam overlay appears but no sound.

**Fixes:**
- User must interact with the page first (click anywhere) before AudioContext can play sound
- Check browser audio is not muted
- Check `_playJamBeep()` is being called (add `console.log`)

---

## Contact

For issues or questions, contact the development team or file an issue in the project repository.
