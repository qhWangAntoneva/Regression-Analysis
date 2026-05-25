#!/bin/bash
# ===========================================================================
# deploy.sh — Deploy Regression Analysis web app to GitHub Pages
# ===========================================================================
#
# Usage:
#   bash web/deploy.sh
#
# This script:
#   1. Checks that all required web assets exist
#   2. Pushes the web/ directory to qhWangAntoneva.github.io repo
#      under /regression-analysis/
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$PROJECT_ROOT/web"
PAGES_DIR="/tmp/gh-pages-deploy-$$"
PAGES_REPO="qhWangAntoneva/qhWangAntoneva.github.io"
TARGET_SUBDIR="regression-analysis"

echo "================================================"
echo "  Regression Analysis — GitHub Pages Deploy"
echo "================================================"
echo ""

# --- Check gh CLI ---
if ! command -v gh &> /dev/null; then
    echo "ERROR: gh (GitHub CLI) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# --- Check web assets ---
echo "[1/5] Checking web assets..."
REQUIRED_FILES=(
    "$WEB_DIR/index.html"
    "$WEB_DIR/css/styles.css"
    "$WEB_DIR/js/app.js"
    "$WEB_DIR/js/gallery_data.js"
    "$WEB_DIR/py/bridge.py"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done
echo "  All required files found."

# --- Clone pages repo ---
echo "[2/5] Cloning pages repo..."
rm -rf "$PAGES_DIR"
git clone "https://github.com/$PAGES_REPO.git" "$PAGES_DIR" --depth 1

# --- Copy web assets ---
echo "[3/5] Copying web assets to $TARGET_SUBDIR/ ..."
TARGET="$PAGES_DIR/$TARGET_SUBDIR"
rm -rf "$TARGET"
mkdir -p "$TARGET"

# Copy all web files
cp -r "$WEB_DIR"/* "$TARGET/" 2>/dev/null || true

# The deploy.sh itself should not be deployed
rm -f "$TARGET/deploy.sh"

echo "  Files copied to $TARGET/"

# --- Commit and push ---
echo "[4/5] Committing and pushing..."
cd "$PAGES_DIR"
git add "$TARGET_SUBDIR/"
if git diff --cached --quiet; then
    echo "  No changes to deploy."
else
    git config user.name "Deploy Bot"
    git config user.email "deploy@regression-analysis.local"
    git commit -m "deploy: regression-analysis web app update

Source: qhWangAntoneva/Regression-Analysis
$(date -Iseconds)"
    git push
    echo "  Deployed successfully!"
fi

# --- Cleanup ---
echo "[5/5] Cleaning up..."
cd "$PROJECT_ROOT"
rm -rf "$PAGES_DIR"

echo ""
echo "================================================"
echo "  Deployment complete!"
echo "  https://${PAGES_REPO%.*}.github.io/$TARGET_SUBDIR/"
echo "================================================"
