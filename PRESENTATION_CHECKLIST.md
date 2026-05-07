# 🎤 CogniRad Presentation Checklist

## 📅 One Week Before

- [ ] Choose deployment platform (Render recommended)
- [ ] Deploy application
- [ ] Test deployment URL
- [ ] Generate QR code with your deployment URL
- [ ] Add QR code to presentation slides

## 📅 One Day Before

- [ ] Test the deployed app end-to-end
  - [ ] Student login works
  - [ ] Chat messages send/receive
  - [ ] Admin dashboard loads
  - [ ] Channel reallocation triggers
- [ ] Verify QR code scans correctly (test with phone)
- [ ] Prepare backup plan (local deployment or ngrok)
- [ ] Charge laptop and phone

## ⏰ 2 Hours Before

- [ ] Wake up the app (if using Render free tier)
  - Open: `https://your-app.onrender.com/`
  - Wait for it to load completely
- [ ] Test login with a dummy CMS ID
- [ ] Clear any test data if needed
- [ ] Verify admin dashboard works
- [ ] Check internet connection at venue

## ⏰ 15 Minutes Before

- [ ] Open admin dashboard in browser: `https://your-app.com/admin.html`
- [ ] Keep this tab open (don't close it)
- [ ] Test one student login to confirm everything works
- [ ] Have backup plan ready (local server or ngrok)
- [ ] Ensure projector/screen is working

## 🎬 During Presentation

### Slide 1: Problem Statement
- [ ] Explain WiFi congestion problem
- [ ] Mention cognitive radio concept
- [ ] Set up the demo context

### Slide 2: QR Code
- [ ] Display QR code prominently
- [ ] Say: "Scan this QR code to join the demo"
- [ ] Give URL verbally as backup: "Or visit [your-url].com"
- [ ] Wait 30 seconds for students to join

### Slide 3: Admin Dashboard (Project This)
- [ ] Switch to admin dashboard tab
- [ ] Point out the 5 channels
- [ ] Show students being distributed
- [ ] Explain channel status colors:
  - 🟢 FREE = Available
  - 🟡 BUSY = Active but healthy
  - 🟠 CONGESTED = Overloaded
  - 🔴 JAMMED = Critical

### Slide 4: Live Demo - Congestion
- [ ] Ask students to send messages rapidly
- [ ] Point out channel status changing
- [ ] Show automatic reallocation happening
- [ ] Explain: "Watch how the system moves students to balance load"

### Slide 5: Manual Control
- [ ] Use admin panel to force-jam a channel
- [ ] Show students getting reallocated
- [ ] Explain minimum-move algorithm
- [ ] Highlight round-robin fairness

### Slide 6: Technical Deep Dive
- [ ] Show dynamic threshold scaling formula
- [ ] Explain sqrt(N) scaling rationale
- [ ] Mention PHY layer simulation
- [ ] Discuss real-world applications

### Slide 7: Results & Conclusion
- [ ] Summarize what was demonstrated
- [ ] Show key metrics (if collected)
- [ ] Mention future work
- [ ] Thank students for participating

## 🎯 Key Talking Points

### When showing QR code:
> "I've deployed a live cognitive radio system. Scan this QR code to join as a student. You'll be automatically assigned to a channel based on current load."

### When showing admin dashboard:
> "This is the control plane. Each channel represents a WiFi frequency band. Watch as students join and the system distributes them intelligently."

### When triggering congestion:
> "Now let's stress test it. Everyone send messages rapidly. Notice how channels turn orange, then red? The system is detecting congestion and automatically reallocating students to healthier channels."

### When manually jamming:
> "I can also manually jam a channel to simulate interference. Watch how the system evacuates all students from the jammed channel and redistributes them fairly using round-robin selection."

### Technical highlight:
> "The thresholds aren't fixed. They scale dynamically with the square root of the number of users, which mirrors how real WiFi handles contention. This means the system works correctly whether you have 1 user or 50."

## 🚨 Emergency Responses

### "The app is slow/not loading"
> "The free tier sleeps after inactivity. Give it 30 seconds to wake up. This is why I opened it before the presentation started."

### "I can't connect"
> "Make sure you're using HTTPS, not HTTP. If campus WiFi is blocking it, try mobile data."

### "WebSocket connection failed"
> "Some networks block WebSockets. The app has a REST fallback, so messages will still work, just with a slight delay."

### "Nothing is happening"
> "Let me check the admin dashboard... [troubleshoot live]. This is actually a great teaching moment about distributed systems and network reliability."

## 📊 Metrics to Highlight

If you have time to collect data during demo:

- [ ] Number of students who joined
- [ ] Number of reallocations triggered
- [ ] Average channel utilization
- [ ] Response time for reallocation
- [ ] Number of messages sent

## 📸 Post-Presentation

- [ ] Take screenshot of admin dashboard with activity
- [ ] Export any demo data
- [ ] Thank students who participated
- [ ] Share deployment URL for continued testing
- [ ] Add recording/screenshots to portfolio

## 🎁 Bonus Points

### If time permits:
- [ ] Show spectrum visualization: `https://your-app.com/spectrum.html`
- [ ] Demonstrate terminal dashboard (if running locally)
- [ ] Show code snippets (allocator.py, classifier.py)
- [ ] Explain ML model (if relevant to audience)

### For technical audience:
- [ ] Mention FastAPI + WebSocket architecture
- [ ] Discuss SQLite for state persistence
- [ ] Explain decay algorithms
- [ ] Show PHY layer simulation code

### For non-technical audience:
- [ ] Use WiFi congestion analogy (coffee shop, airport)
- [ ] Compare to traffic management systems
- [ ] Emphasize real-world applications (5G, IoT)

## ✅ Success Criteria

Your demo is successful if:
- [ ] Students can join via QR code
- [ ] Messages send and receive in real-time
- [ ] Admin dashboard shows live updates
- [ ] Automatic reallocation triggers and completes
- [ ] Manual jam command works
- [ ] Audience understands the concept

## 🎉 You've Got This!

Remember:
- **Breathe** - You know this system inside and out
- **Engage** - Make eye contact, ask questions
- **Adapt** - If something breaks, explain why (it's a learning moment)
- **Enjoy** - This is your moment to shine!

Good luck! 🚀
