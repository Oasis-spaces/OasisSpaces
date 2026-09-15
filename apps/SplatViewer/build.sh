#!/bin/bash
# Builds "Splat Viewer.app" into apps/SplatViewer/build/.
#   ./build.sh            build
#   ./build.sh --install  build, then copy the app to ~/Applications
set -euo pipefail
cd "$(dirname "$0")"

command -v xcodegen >/dev/null || { echo "Needs XcodeGen: brew install xcodegen"; exit 1; }
xcodegen generate --quiet
log=build/xcodebuild.log
mkdir -p build
xcodebuild -project SplatViewer.xcodeproj -scheme SplatViewer -configuration Release \
    -derivedDataPath build/DerivedData -destination 'platform=macOS,arch=arm64' \
    build > "$log" 2>&1 || true
grep -E "error:|warning: .*Sources/" "$log" | sort -u || true
if ! grep -q "BUILD SUCCEEDED" "$log"; then
    echo "BUILD FAILED (full log: $(pwd)/$log)"
    exit 1
fi

app="build/DerivedData/Build/Products/Release/Splat Viewer.app"
rm -rf "build/Splat Viewer.app"
cp -R "$app" "build/Splat Viewer.app"
echo "Built: $(pwd)/build/Splat Viewer.app"

if [ "${1:-}" = "--install" ]; then
    mkdir -p ~/Applications
    rm -rf ~/Applications/"Splat Viewer.app"
    cp -R "build/Splat Viewer.app" ~/Applications/
    echo "Installed: ~/Applications/Splat Viewer.app"
fi
