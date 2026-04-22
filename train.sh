set -e

CONFIG=${1:-configs/train.yaml}

echo "============================================================"
echo "Banking Intent Classification — Training Pipeline"
echo "Config: $CONFIG"
echo "============================================================"

echo ""
echo "[Step 1] Preprocessing data..."
python scripts/preprocess_data.py --config "$CONFIG"

echo ""
echo "[Step 2] Training model..."
python scripts/train.py --config "$CONFIG"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
