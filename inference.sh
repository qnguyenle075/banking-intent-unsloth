set -e

CONFIG="configs/inference.yaml"

if [ -n "$1" ]; then
    python scripts/inference.py --config "$CONFIG" --message "$1"
else
    python scripts/inference.py --config "$CONFIG"
fi
