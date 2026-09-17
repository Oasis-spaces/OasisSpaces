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
xcodegen generate --quiet

# The first paired iPhone that is connected: CoreDevice id for devicectl, UDID for xcodebuild.
# ("unavailable" contains "available": match the connected state exactly.)
device=$(xcrun devicectl list devices 2>/dev/null | awk '/iPhone/ && / available \(paired\)/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9A-F-]{36}$/) {print $i; exit}}')
[ -n "$device" ] || { echo "No connected iPhone (unlock it and check it is paired)"; exit 1; }
udid=$(xcrun devicectl device info details --device "$device" 2>/dev/null | awk '/ udid:/ {print $NF; exit}')

log=build/xcodebuild.log
mkdir -p build
xcodebuild -project OasisCapture.xcodeproj -scheme OasisCapture -configuration Debug \
    -destination "id=$udid" -derivedDataPath build/DerivedData -allowProvisioningUpdates \
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
