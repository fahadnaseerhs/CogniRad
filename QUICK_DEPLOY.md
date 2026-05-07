# 🚀 Quick Deploy for Presentation

## TL;DR - Get Your App Live in 10 Minutes

### Step 1: Choose Platform (Pick One)

#### ⭐ **Render.com** (Easiest - Recommended)
```bash
# 1. Push to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Go to render.com → Sign up
# 3. New + → Web Service → Connect GitHub repo
# 4. Auto-detects render.yaml → Create
# 5. Wait 5 minutes → Get URL
```

#### 🚂 **Railway.app** (Fastest)
```bash
npm install -g @railway/cli
railway login
railway init
railway up
railway domain  # Get your URL
```

#### ✈️ **Fly.io** (Global)
```bash
# Windows PowerShell:
iwr https://fly.io/install.ps1 -useb | iex

# Then:
fly auth login
fly launch --now
```

---

### Step 2: Generate QR Code

```bash
# Install QR library
pip install qrcode[pil]

# Generate QR code (replace with your URL)
python generate_qr.py https://your-app.onrender.com
```

This creates `cognirad_qr.png` - add it to your presentation slide!

---

### Step 3: Test Before Presentation

```bash
# Health check
curl https://your-app-url.com/

# Test login
curl -X POST https://your-app-url.com/login \
  -H "Content-Type: application/json" \
  -d '{"cms": "TEST001"}'
```

---

### Step 4: Presentation Setup

**2 minutes before presenting:**

1. **Wake the app** (if using Render free tier):
   - Open `https://your-app-url.com/` in browser
   - Wait for it to load (first load after sleep takes ~30 sec)

2. **Open admin dashboard** (for projection):
   - URL: `https://your-app-url.com/admin.html`
   - Keep this tab open to show real-time allocation

3. **Display QR code** on slide:
   - Students scan → Join with CMS ID
   - Watch them appear on admin dashboard

---

## Presentation Flow

### Slide 1: Introduction
- Show the problem: WiFi congestion in crowded spaces
- Explain cognitive radio concept

### Slide 2: Live Demo - QR Code
```
┌─────────────────────────────────┐
│  Scan to Join CogniRad Demo     │
│                                 │
│   [QR CODE IMAGE HERE]          │
│                                 │
│  Or visit: your-app-url.com     │
└─────────────────────────────────┘
```

### Slide 3: Admin Dashboard (Project This)
- Show real-time channel states
- Students joining → automatic distribution
- Channel status: FREE → BUSY → CONGESTED

### Slide 4: Trigger Congestion
- Ask students to send messages rapidly
- Show automatic reallocation in action
- Highlight round-robin fairness

### Slide 5: Manual Control
- Use admin panel to force-jam a channel
- Watch students get reallocated
- Show minimum-move algorithm

---

## URLs You Need

| Purpose | URL |
|---------|-----|
| Student App | `https://your-app.com/` |
| Admin Dashboard | `https://your-app.com/admin.html` |
| Spectrum View | `https://your-app.com/spectrum.html` |
| API Health | `https://your-app.com/channel/state` |

---

## Troubleshooting

### App is sleeping (Render free tier)
**Solution**: Open the URL 2 minutes before presentation

### Students can't connect
- Check URL is correct (https://, not http://)
- Test on mobile data (campus WiFi might block)
- Verify CORS is enabled (already configured)

### WebSocket errors
- Ensure using `wss://` (secure WebSocket)
- Check browser console for errors
- Some networks block WebSockets (use mobile hotspot)

### Database resets
- Normal on free tiers
- For persistent data, upgrade to paid tier
- Or use external database (not needed for demo)

---

## Cost Breakdown

| Platform | Free Tier | Paid | Notes |
|----------|-----------|------|-------|
| **Render** | ✅ Yes (sleeps) | $7/mo | Best for demo |
| **Railway** | $5 credit | ~$5-10/mo | Fast deploys |
| **Fly.io** | 3 VMs free | Pay-as-go | Global edge |

**Recommendation**: Use Render free tier for your presentation. It's perfect for a one-time demo.

---

## After Presentation

### Keep it running
- Share URL with professors/peers
- Add to resume as live project
- Include in portfolio

### Export demo data
```bash
# Download database (if needed)
railway run bash
# or Render SSH
```

### Share recording
- Record screen during demo
- Show QR → students joining → reallocation
- Upload to YouTube/portfolio

---

## Emergency Backup Plan

If deployment fails before presentation:

### Option 1: Local Network Demo
```bash
# Run locally
uvicorn main:app --host 0.0.0.0 --port 8000

# Get your local IP
ipconfig  # Windows
ifconfig  # Mac/Linux

# Students connect to: http://YOUR-IP:8000
```

### Option 2: ngrok Tunnel
```bash
# Install ngrok
choco install ngrok  # Windows
brew install ngrok   # Mac

# Run app
uvicorn main:app --port 8000

# In another terminal
ngrok http 8000

# Use the ngrok URL (https://xxxx.ngrok.io)
```

---

## Questions?

See full deployment guide: `DEPLOYMENT.md`

**Good luck with your presentation! 🎉**
