# DESK — Vinkona's desktop face

The consolidation-phase GUI: the config panel reborn as a friendly desktop app for a
non-technical audience, with the machinery hidden but reachable, shipping as an AppImage
on Linux (native equivalents elsewhere).  Lives in `desk/` (Flutter/Dart).

## Why Dart/Flutter (decision record, 2026-08-17)

* One framework across Android (the existing `vinkona_simi` client), Linux, macOS,
  Windows — one language, shared widgets later, and the toolchain is already proven on
  the dev box (arm64 APKs build green).
* Flutter's Linux build is a self-contained GTK bundle with **no webview dependency** —
  the make-or-break for reliable AppImages.  The Tauri launcher (P1) depended on system
  webkit2gtk (the classic brittle-AppImage cause) and a Rust toolchain the dev sandbox
  can't run; it is **retired** by this plan, and its one load-bearing idea — the
  `supervisor status --json` seam — is absorbed here (stage D3).
* Electron: heavy, no mobile story.  Qt/PySide: consumer styling costs more, weak mobile
  story, forks the GUI investment away from the phone client.

## The rules

1. **Thin client, always.**  The app is a pure client of the backend's localhost HTTP
   API (config_server + cascade).  No decision logic in Dart — audience tiers
   (FIELD_LEVELS), preview-then-confirm stories (FEATURE_RECIPES), activity resolution,
   doctor probes all stay server-side where the web panel and phone client share them.
   If a screen needs something the API lacks, extend the API, then render it.
2. **The web panel survives** as the expert/fallback surface.  The desktop app is the
   consumer face over the same seams; nothing is ported away from the panel.
3. **Basic first.**  Every surface ships its Basic form before its Advanced form.
   Advanced/expert live behind ONE deliberate gate per screen, never mixed in.
4. **Config writes are whole-object** (`GET /api/config` merged → mutate → `POST`),
   matching the web panel.  A keyed setter endpoint is a later backend nicety, not a
   blocker.

## Stages

Each stage is a committable slice with its own acceptance test; later stages never
block earlier ones from shipping.

* **D0 — toolchain** (Dan): `sudo dnf install clang cmake ninja-build gtk3-devel`
  on the dev VM, then `flutter doctor` shows Linux desktop green.
  *Accept: `flutter build linux` produces a runnable bundle of the D1 skeleton.*

* **D1 — skeleton** (DONE, this commit): `desk/` app; `BackendClient` (config,
  field_levels, feature_recipes, help, activity, whole-config save); shell with
  navigation (Home / Settings / Tools / Live); Home = connection + live activity card
  (3s poll, graceful when the backend is down); Settings-Basic = the 16 FEATURE_RECIPES
  as plain-language switches with the preview-then-confirm sheet (companion changes
  listed, nothing saved before confirm); stubs are honest about what's coming.
  *Accept: `flutter analyze` clean; widget tests prove flip→preview→confirm posts the
  switch + companions in one save, and cancel changes nothing.  (8 tests green.)*

* **D2 — Basic settings, complete**: the remaining Basic-tier knobs that aren't recipe
  switches (persona picker via `/api/profiles` + `/api/personas`, TTS voice via
  `/api/tts` + `/api/tts/select`, awareness basics), grouped in friendly sections with
  `/api/help` text as subtitles.  First-run state: backend not installed → point at the
  installer instead of an error.
  *Accept: a non-technical user can change everything Basic without seeing a dotted
  path or JSON.*

* **D3 — Home becomes a real dashboard**: services health (`/api/services`,
  supervisor seam), doctor-style rows with the one-line fix when something's down,
  idle pause/resume (`/api/idle`), restart button (`/api/restart`), and the activity
  pill wired to the same states the web header shows.
  *Accept: backend stopped/degraded states all render with a next step, never a stack
  trace.*

* **D4 — the machinery, gated**: Advanced settings (full config tree generated from
  merged config + FIELD_LEVELS, exactly like the web form, behind the gate); Tools
  (roster + usage, the ideas/spec queue with statuses, inspect-attempt editor,
  Run-toolsmith-now); Live (trace events + the Big/Fast LM feeds incl. the
  streaming-now block via `/api/lm_feed`).
  *Accept: feature parity with the web panel's Tools + Live tabs; an expert never
  NEEDS the browser (but may still prefer it).*

* **D5 — chat surface**: text chat with the cascade (its token-gated HTTP routes),
  desktop-adaptive layout.  This is also where shared widgets get extracted into a
  package the phone client (`vinkona_simi`) can import — the reunification point.
  *Accept: a conversation held entirely from the desktop app.*

* **D6 — packaging**: `flutter_distributor` → AppImage.  The AppImage carries the GUI +
  the supervisor/bootstrap glue (and may carry the prebuilt llama-server); it must NOT
  carry model weights (user data, `~/.local/share/vinkona` via the existing install.sh
  machinery — the LM Studio on-ramp flow) and CANNOT carry podman (own-tools keep using
  host podman/bwrap; the doctor row says so).  First run offers the provisioning flow.
  macOS `.dmg` / Windows installer from the same distributor config when the
  platform-independence Windows pass lands.
  *Accept: a fresh Linux box goes AppImage-download → first-run setup → talking to
  Vinkona without touching a terminal.*

## Known dependencies / risks

* The panel API is localhost-only by design (`_local_only`).  Right for the desktop
  app (same box).  When the PHONE wants these surfaces (post-D5), the API needs a
  token-auth story instead of a LAN bind — decide then, not now.
* Whole-config POST freezes merged defaults into config.json (same as the web panel
  today).  Acceptable; a keyed setter would make saves surgical — backend nicety.
* Flutter pins: dev box currently Flutter 3.44.x / Dart ^3.12; keep `desk/` and
  `vinkona_simi` on the same channel to ease the D5 package extraction.
