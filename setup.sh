#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== QuestClass 3D Backend Setup ==="

# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Clone SAM 3D Objects (Meta)
if [ ! -d "sam-3d-objects" ]; then
  echo "Cloning SAM 3D Objects..."
  git clone https://github.com/facebookresearch/sam-3d-objects.git
fi
cd sam-3d-objects
pip install -r requirements.txt
pip install -r requirements.inference.txt
cd ..

# 4. Download SAM checkpoint (ViT-H, ~2.4 GB)
CKPT="sam_vit_h_4b8939.pth"
if [ ! -f "$CKPT" ]; then
  echo "Downloading SAM ViT-H checkpoint..."
  wget -q --show-progress \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
fi

# 5. Download SAM 3D checkpoint (follow instructions in sam-3d-objects/README)
echo ""
echo "=== IMPORTANT ==="
echo "Download SAM 3D checkpoints from the model page and place in:"
echo "  $SCRIPT_DIR/sam-3d-objects/checkpoints/"
echo ""
echo "Then copy your Firebase service account key to:"
echo "  $SCRIPT_DIR/serviceAccountKey.json"
echo ""
echo "Copy .env.example to .env and fill in your values."
echo ""
echo "Setup complete! Run: source .venv/bin/activate && uvicorn main:app --reload --port 8000"
