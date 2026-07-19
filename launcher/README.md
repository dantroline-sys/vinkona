# Vinkona launcher

A **thin desktop shell** over the stack: double-clickable on macOS (`.app`)
and Linux (a single native binary + `.desktop` entry), showing live service
status with the supervisor's reason lines, Start/Stop/Restart buttons, a
tray icon, and the two web UIs (Settings `:8090`, Knowledge panel `:8771`)
hosted in native webview windows.

Principle: the launcher is **never a second orchestrator**. Every button
shells out to `vinkona.sh` / `assistant/supervisor.py` and every row renders
`supervisor.py status --json`. Close the launcher and nothing changes;
`./vinkona.sh` stays fully equivalent.

## Building

Tauri v2 uses the system webview (WKWebView / WebKitGTK) — no Node, no
bundled Chromium; the frontend is the static page in `ui/`.

**Linux (Fedora):**

```bash
sudo dnf install gtk3-devel webkit2gtk4.1-devel libappindicator-gtk3-devel \
                 librsvg2-devel openssl-devel
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # once
cd launcher/src-tauri
cargo build --release              # -> target/release/vinkona-launcher
# packaged formats (AppImage/rpm/deb):
cargo install tauri-cli --locked && cargo tauri build
```

**macOS:** Xcode CLT + rustup, then the same `cargo tauri build` →
`Vinkona.app` (and a .dmg). Unsigned is fine for personal machines
(right-click → Open the first time); signing/notarization is deferred until
distribution matters.

## How it finds the stack

Stored choice → `$VINKONA_DIR` → ancestors of the executable (dev builds
inside the repo find it automatically) → `~/vinkona`. First run outside
those: the **pick folder** dialog, persisted in the app config dir.

## Layout

```
launcher/
  ui/index.html        the page (status table, buttons) — plain HTML/JS
  make_icons.py        stdlib icon generator (PNG/ICO/ICNS) — output committed
  src-tauri/
    src/main.rs        commands: get_state/pick_checkout/status/action/open_ui + tray
    tauri.conf.json    static frontendDist, window, bundle icons
    capabilities/      Tauri v2 ACL: core:default + dialog:default only
```

## Roadmap (P2/P3)

- Front-end `install.sh` (it's already an idempotent checklist — drive it
  with visible progress) and `fetch_models.sh` (small/full as a dialog).
- Start-at-login (autostart plugin), log viewer window.
- dmg/AppImage polish, first-run onboarding.
