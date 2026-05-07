# CogniRad Deployment Guide

## Quick Deploy Options for Live Presentation

### Option 1: Render.com (Recommended for Demo)

**Why**: Free tier, auto-SSL, WebSocket support, simple setup

**Steps**:
1. Push code to GitHub:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. Go to [render.com](https://render.com) and sign up

3. Click "New +" → "Web Service"

4. Connect your GitHub repo

5. Render auto-detects the `render.yaml` config:
   - **Name**: cognirad
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

6. Click "Create Web Service"

7. Wait 3-5 minutes for deployment

8. Your URL: `https://cognirad-XXXX.onrender.com`

**Generate QR Code**:
```bash
# Install qrcode library
pip install qrcode[pil]

# Generate QR code
python -c "import qrcode; qr = qrcode.QRCode(); qr.add_data('https://YOUR-APP.onrender.com'); qr.make(); img = qr.make_image(); img.save('cognirad_qr.png')"
```

**Important Notes**:
- Free tier sleeps after 15 min inactivity
- First request after sleep takes ~30 seconds to wake
- **Solution**: Hit the URL 2 minutes before your presentation starts
- Or upgrade to $7/month for always-on

---

### Option 2: Railway.app

**Why**: $5 free credit, fast deploys, great DX

**Steps**:
1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   # or
   curl -fsSL https://railway.app/install.sh | sh
   ```

2. Login and deploy:
   ```bash
   railway login
   railway init
   railway up
   ```

3. Get your URL:
   ```bash
   railway domain
   ```

4. Your app is live at `https://cognirad-production.up.railway.app`

**Cost**: Free $5 credit/month, then ~$5-10/month

---

### Option 3: Fly.io

**Why**: Free tier, global edge network, fast

**Steps**:
1. Install Fly CLI:
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   
   # Mac/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. Login and launch:
   ```bash
   fly auth login
   fly launch
   ```

3. Follow prompts:
   - App name: `cognirad`
   - Region: Choose closest to your presentation location
   - Database: No
   - Deploy now: Yes

4. Your URL: `https://cognirad.fly.dev`

**Cost**: Free tier includes 3 VMs (sufficient for demo)

---

### Option 4: PythonAnywhere

**Why**: Education-focused, web interface

**Steps**:
1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)

2. Upload your code via Files tab or Git:
   ```bash
   git clone https://github.com/YOUR-USERNAME/cognirad.git
   ```

3. Create virtual environment:
   ```bash
   mkvirtualenv cognirad --python=python3.11
   pip install -r requirements.txt
   ```

4. Configure Web App:
   - Web tab → Add new web app
   - Manual configuration → Python 3.11
   - WSGI file: Point to your app
   - Static files: `/static/` → `/home/USERNAME/cognirad/static/`

5. Your URL: `https://USERNAME.pythonanywhere.com`

**Limitation**: Free tier has WebSocket restrictions (may need paid tier for chat)

---

## Pre-Presentation Checklist

### 1. Test the Deployment
```bash
# Health check
curl https://YOUR-APP-URL.com/

# Test login
curl -X POST https://YOUR-APP-URL.com/login \
  -H "Content-Type: application/json" \
  -d '{"cms": "CMS001"}'
```

### 2. Generate QR Code for Presentation

**Online Tool** (easiest):
- Go to [qr-code-generator.com](https://www.qr-code-generator.com/)
- Paste your deployment URL
- Download PNG

**Python Script**:
```python
import qrcode

# Your deployed URL
url = "https://cognirad.onrender.com"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("cognirad_qr_code.png")
print(f"QR code saved! Students scan to access: {url}")
```

### 3. Prepare Demo Data

**Pre-load some students** (optional):
```bash
# Add test students before presentation
curl -X POST https://YOUR-APP-URL.com/login -H "Content-Type: application/json" -d '{"cms": "CMS001"}'
curl -X POST https://YOUR-APP-URL.com/login -H "Content-Type: application/json" -d '{"cms": "CMS002"}'
```

### 4. Admin Dashboard Access

- Admin URL: `https://YOUR-APP-URL.com/admin.html`
- Project this during presentation to show real-time channel states
- Students use: `https://YOUR-APP-URL.com/` (main app)

### 5. Wake Up the App (Render Free Tier)

**2 minutes before presentation**:
```bash
# Hit the URL to wake it from sleep
curl https://YOUR-APP-URL.onrender.com/
```

Or open it in a browser tab and keep it open.

---

## Presentation Flow

### Slide 1: Show QR Code
- Display QR code on screen
- Students scan and join with their CMS ID

### Slide 2: Admin Dashboard
- Project `https://YOUR-APP-URL.com/admin.html`
- Show real-time channel allocation
- Watch as students join and get distributed

### Slide 3: Trigger Congestion
- Ask students to send messages rapidly
- Show channel status changing: FREE → BUSY → CONGESTED
- Demonstrate automatic reallocation

### Slide 4: Manual Jam
- Use admin panel to force-jam a channel
- Show students getting reallocated in real-time

---

## Troubleshooting

### App Won't Start
- Check logs: `railway logs` or Render dashboard
- Verify `requirements.txt` has all dependencies
- Ensure `PORT` environment variable is used

### WebSocket Connection Fails
- Check CORS settings in `main.py`
- Verify WebSocket URL uses `wss://` (not `ws://`)
- Some free tiers limit WebSocket duration

### Database Issues
- SQLite works fine for demos (already configured)
- Database persists between deploys on Render/Railway
- For Fly.io, add a volume if you need persistence

### Students Can't Connect
- Verify URL is accessible: `curl https://YOUR-APP-URL.com/`
- Check firewall/network restrictions
- Test on mobile data (not campus WiFi)

---

## Cost Summary

| Platform       | Free Tier          | Paid Tier       | Best For              |
|----------------|--------------------|-----------------|-----------------------|
| Render         | Yes (sleeps)       | $7/mo always-on | Quick demo            |
| Railway        | $5 credit/mo       | ~$5-10/mo       | Student projects      |
| Fly.io         | 3 VMs free         | Pay-as-you-go   | Global audience       |
| PythonAnywhere | Limited WebSocket  | $5/mo           | Education/learning    |

**Recommendation**: Start with Render free tier for your presentation. Upgrade to $7/month if you need it always-on for multiple demos.

---

## Post-Presentation

### Export Demo Data
```bash
# Download the database
scp YOUR-APP:/path/to/cognirad.db ./cognirad_demo.db
```

### Share Recording
- Record screen during demo
- Show QR code → students joining → reallocation in action
- Add to your portfolio/GitHub

### Keep It Running
- If presentation goes well, keep the deployment live
- Share URL with professors/peers for review
- Add to your resume as a live project link
