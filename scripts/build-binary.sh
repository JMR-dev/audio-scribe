#!/usr/bin/env bash
# Build a self-contained ccn-transcribe directory that runs without Python or uv.
#
# onedir, never onefile: onefile extracts ~450 MB to /tmp on every launch.
# deno lands in bin/ because PyInstaller points sys.prefix at the bundle, which
# is where the package already looks for the JS runtime it ships with.
set -euo pipefail

cd "$(dirname "$0")/.."
DIST=${1:-dist}

uv run --locked --with pyinstaller pyinstaller \
  --noconfirm --onedir --name ccn-transcribe \
  --distpath "$DIST" --workpath "build/pyinstaller" --specpath "build" \
  --collect-all openvino \
  --collect-all openvino_genai \
  --collect-all openvino_tokenizers \
  --collect-all yt_dlp \
  --collect-all yt_dlp_ejs \
  --add-binary "$PWD/.venv/bin/deno:bin" \
  src/ccn_transcribe/__main__.py

echo
echo "built $DIST/ccn-transcribe ($(du -sh "$DIST/ccn-transcribe" | cut -f1))"
echo "verify it with: $DIST/ccn-transcribe/ccn-transcribe doctor"
