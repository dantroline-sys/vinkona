# Vinkona launcher

The **easy-mode face** of the stack: a sleek desktop app (macOS `.app`,
Linux native binary — run `./Vinkona` at the repo root) with a live status
hero, one Start/Stop button, a tray icon, the web UIs in native windows —
and **wizards instead of switches**: first-run setup (model check +
RAM-sized download with live progress), and "Connect a knowledge box"
(URL + token → live probe → config written for you, including the optional
remote big-LM hookup). The config web UI stays the expert surface, one
click away under "All settings".

Principle: the launcher is **never a second orchestrator**. Every button
shells out to `vinkona.sh` / `assistant/supervisor.py`, and every wizard
action is one small, tested supervisor JSON verb (`preflight`,
`vinur-probe`, `config-patch`, `fetch-models`, `logtail`) piped through a
single whitelisted Rust command — the logic lives in Python, the app is a
dumb pipe.  Close the launcher and nothing changes; `./vinkona.sh` stays
fully equivalent.

`./install-desktop.sh` (here) adds Vinkona to the Linux application menu
(a `.desktop` entry pointing at the root `./Vinkona` wrapper, with the
committed icon); `--uninstall` removes it.

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

- DONE: `fetch_models.sh` driven from the setup wizard with live progress;
  connect-a-knowledge-box wizard; `.desktop` entry; root `./Vinkona`.
- Front-end `install.sh` (an idempotent checklist — drive it with visible
  progress) from the wizard too.
- Start-at-login (autostart plugin), log viewer window.
- dmg/AppImage polish.
