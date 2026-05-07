#!/bin/bash
# Quick deployment helper script for CogniRad

set -e

echo "🚀 CogniRad Deployment Helper"
echo "=============================="
echo ""

# Check if git repo exists
if [ ! -d .git ]; then
    echo "❌ Not a git repository. Initializing..."
    git init
    git add .
    git commit -m "Initial commit - CogniRad ready for deployment"
fi

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "📝 Uncommitted changes detected. Committing..."
    git add .
    git commit -m "Deployment preparation - $(date +%Y-%m-%d)"
fi

echo ""
echo "Choose deployment platform:"
echo "1) Render.com (Recommended - Free tier)"
echo "2) Railway.app (Fast - $5 free credit)"
echo "3) Fly.io (Global edge - Free tier)"
echo "4) Manual (Show instructions)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "📦 Render.com Deployment"
        echo "========================"
        echo ""
        echo "Steps:"
        echo "1. Push to GitHub:"
        echo "   git remote add origin https://github.com/YOUR-USERNAME/cognirad.git"
        echo "   git push -u origin main"
        echo ""
        echo "2. Go to https://render.com and sign up"
        echo "3. Click 'New +' → 'Web Service'"
        echo "4. Connect your GitHub repo"
        echo "5. Render will auto-detect render.yaml"
        echo "6. Click 'Create Web Service'"
        echo ""
        echo "Your app will be live at: https://cognirad-XXXX.onrender.com"
        ;;
    2)
        echo ""
        echo "🚂 Railway.app Deployment"
        echo "========================="
        echo ""
        if ! command -v railway &> /dev/null; then
            echo "Installing Railway CLI..."
            npm install -g @railway/cli
        fi
        echo "Deploying to Railway..."
        railway login
        railway init
        railway up
        echo ""
        echo "✓ Deployed! Get your URL:"
        railway domain
        ;;
    3)
        echo ""
        echo "✈️  Fly.io Deployment"
        echo "===================="
        echo ""
        if ! command -v fly &> /dev/null; then
            echo "Installing Fly CLI..."
            curl -L https://fly.io/install.sh | sh
        fi
        echo "Deploying to Fly.io..."
        fly auth login
        fly launch --now
        ;;
    4)
        echo ""
        echo "📖 Manual Deployment Instructions"
        echo "=================================="
        echo ""
        echo "See DEPLOYMENT.md for detailed instructions"
        echo ""
        cat DEPLOYMENT.md
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✓ Deployment process complete!"
echo ""
echo "Next steps:"
echo "1. Test your deployment URL"
echo "2. Generate QR code: python generate_qr.py YOUR-URL"
echo "3. Add QR code to your presentation"
echo ""
