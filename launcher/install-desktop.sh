#!/bin/bash
# Put Vinkona in the Linux application menu: writes a .desktop entry pointing
# at the repo-root ./Vinkona wrapper, with the launcher icon.  Idempotent;
# --uninstall removes it.  (macOS doesn't need this — `cargo tauri build`
# produces Vinkona.app there.)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ENTRY="$APPS/vinkona.desktop"

if [ "$1" = "--uninstall" ]; then
    rm -f "$ENTRY"
    echo "removed $ENTRY"
    exit 0
fi

ICON="$ROOT/launcher/src-tauri/icons/128x128.png"
[ -f "$ICON" ] || ICON="$ROOT/launcher/src-tauri/icons/icon.png"
mkdir -p "$APPS"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=Vinkona
Comment=Local, private voice assistant — status, setup and control
Exec=$ROOT/Vinkona
Icon=$ICON
Terminal=false
Categories=Utility;AudioVideo;
EOF
chmod +x "$ROOT/Vinkona" 2>/dev/null || true
echo "installed $ENTRY (menu entry may need a re-login or 'update-desktop-database')"
