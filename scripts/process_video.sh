#!/bin/bash
# Full OasisSpaces chain: video -> cloud -> dense -> shapes -> Blender -> splat
# Usage: scripts/process_video.sh <video> <space-name> [fps]
set -e -o pipefail
cd "$(dirname "$0")/.."

VIDEO="$1"
NAME="$2"
FPS="${3:-2}"
SPACE="spaces/$NAME"
export PYTHONUNBUFFERED=1

echo "=== [1/6] Reconstruct (COLMAP) ==="
python3 pipeline/reconstruct.py "$VIDEO" --name "$NAME" --fps "$FPS"

echo "=== [2/6] Neural densify ==="
python3 pipeline/densify.py "$SPACE"

echo "=== [3/6] Shape detection ==="
python3 pipeline/shapes.py "$SPACE"

echo "=== [4/6] Shape classification ==="
python3 tools/classify_shapes.py "$SPACE"

echo "=== [5/6] Blender scene ==="
/Applications/Blender.app/Contents/MacOS/Blender --background \
    --python tools/blender_room.py -- \
    "$SPACE/shapes.json" "$SPACE/room.blend" "$SPACE/room-render.png" \
    | grep -E "saved|rendered" || true

echo "=== [6/6] Gaussian splat ==="
python3 pipeline/splat_seed.py "$SPACE"
tools/opensplat "$SPACE/splat-project" -n 10000 -d 4 \
    -o "$PWD/$SPACE/splat.ply" | tail -3

echo "CHAIN DONE: $SPACE (cloud.ply, cloud-dense.ply, room.blend, splat.ply)"
