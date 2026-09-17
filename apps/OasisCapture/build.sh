#!/bin/bash
# Builds Oasis Capture and runs it on the connected iPhone.
#   ./build.sh              build, install and launch on the first connected iPhone
#   ./build.sh --build-only build for a device without installing
set -euo pipefail
cd "$(dirname "$0")"

command -v xcodegen >/dev/null || { echo "Needs XcodeGen: brew install xcodegen"; exit 1; }
# The on-device segmentation model is generated, not checked in (see scripts/convert_segmentation.py).
if [ ! -d Resources/RoomSegmentation.mlpackage ]; then
    [ -x ~/.venvs/oasis-coreml/bin/python ] || { echo "Needs ~/.venvs/oasis-coreml (Python 3.12 with torch 2.5, transformers 4.46, coremltools 8.3) to build the model"; exit 1; }
    ~/.venvs/oasis-coreml/bin/python scripts/convert_segmentation.py
fi
# The depth model is Apple's Core ML build of Depth Anything V2 Small (48 MB, not checked in).
if [ ! -f Resources/RoomDepth.mlpackage/Data/com.apple.CoreML/weights/weight.bin ]; then
    echo "Downloading the depth model..."
    B=https://huggingface.co/apple/coreml-depth-anything-v2-small/resolve/main/DepthAnythingV2SmallF16.mlpackage
    D=Resources/RoomDepth.mlpackage
    mkdir -p "$D/Data/com.apple.CoreML/weights"
    curl -sL -o "$D/Manifest.json" "$B/Manifest.json"
    curl -sL -o "$D/Data/com.apple.CoreML/model.mlmodel" "$B/Data/com.apple.CoreML/model.mlmodel"
    curl -sL -o "$D/Data/com.apple.CoreML/weights/weight.bin" "$B/Data/com.apple.CoreML/weights/weight.bin"
fi
# The object detector (YOLOE prompted with object-classes.json) is generated too (see scripts/convert_objects.py).
if [ ! -d Resources/RoomObjects.mlpackage ]; then
    ~/.venvs/oasis-coreml/bin/python scripts/convert_objects.py
fi
xcodegen generate --quiet

# The first paired iPhone that is connected: CoreDevice id for devicectl, UDID for xcodebuild.
# ("unavailable" contains "available": match the connected state exactly.)
device=$(xcrun devicectl list devices 2>/dev/null | awk '/iPhone/ && / available \(paired\)/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9A-F-]{36}$/) {print $i; exit}}')
if [ -n "$device" ]; then
    udid=$(xcrun devicectl device info details --device "$device" 2>/dev/null | awk '/ udid:/ {print $NF; exit}')
    destination="id=$udid"
elif [ "${1:-}" = "--build-only" ]; then
    destination="generic/platform=iOS"
else
    echo "No connected iPhone (unlock it and check it is paired)"; exit 1
fi

log=build/xcodebuild.log
mkdir -p build
xcodebuild -project OasisCapture.xcodeproj -scheme OasisCapture -configuration Debug \
    -destination "$destination" -derivedDataPath build/DerivedData -allowProvisioningUpdates \
    build > "$log" 2>&1 || true
grep -E "error:|warning: .*Sources/" "$log" | sort -u || true
if ! grep -q "BUILD SUCCEEDED" "$log"; then
    echo "BUILD FAILED (full log: $(pwd)/$log)"
    exit 1
fi
app=build/DerivedData/Build/Products/Debug-iphoneos/OasisCapture.app
echo "Built: $(pwd)/$app"
[ "${1:-}" = "--build-only" ] && exit 0

xcrun devicectl device install app --device "$device" "$app"
xcrun devicectl device process launch --device "$device" com.oasisspaces.capture
