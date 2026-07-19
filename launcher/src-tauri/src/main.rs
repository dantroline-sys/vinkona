// Vinkona launcher — a THIN desktop shell over the existing stack.
//
// Principle: this app is never a second orchestrator.  It shells out to
// vinkona.sh / assistant/supervisor.py (the one brain) and renders their
// answers; if the launcher dies, nothing about the running stack changes.
//
// Commands the UI invokes:
//   get_state       -> stored checkout path + validity
//   pick_checkout   -> native folder dialog, persisted in the app config dir
//   status          -> `python3 assistant/supervisor.py status --json`
//   action          -> `bash vinkona.sh start|stop|restart`
//   open_ui         -> a webview window on the config UI / knowledge panel
//   supervisor_json -> the wizards' pipe: a whitelisted supervisor.py JSON
//                      verb (preflight / vinur-probe / config-patch /
//                      fetch-models / logtail).  All wizard LOGIC lives in
//                      supervisor.py (tested Python); this stays a dumb pipe.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager, State, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;

#[derive(Serialize, Deserialize, Clone, Default)]
struct LauncherCfg {
    checkout: Option<String>,
}

struct Stored(Mutex<LauncherCfg>);

fn cfg_file(app: &AppHandle) -> PathBuf {
    app.path()
        .app_config_dir()
        .expect("no config dir")
        .join("launcher.json")
}

fn load_cfg(app: &AppHandle) -> LauncherCfg {
    fs::read_to_string(cfg_file(app))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_cfg(app: &AppHandle, cfg: &LauncherCfg) {
    let p = cfg_file(app);
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(p, serde_json::to_string_pretty(cfg).unwrap_or_default());
}

fn looks_like_checkout(p: &Path) -> bool {
    p.join("vinkona.sh").is_file() && p.join("assistant").join("supervisor.py").is_file()
}

/// Stored path, $VINKONA_DIR, ancestors of the executable (dev runs from
/// launcher/src-tauri/target/*), then ~/vinkona — first hit wins.
fn find_checkout(stored: &Option<String>) -> Option<PathBuf> {
    if let Some(s) = stored {
        let p = PathBuf::from(s);
        if looks_like_checkout(&p) {
            return Some(p);
        }
    }
    if let Ok(env) = std::env::var("VINKONA_DIR") {
        let p = PathBuf::from(env);
        if looks_like_checkout(&p) {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors() {
            if looks_like_checkout(anc) {
                return Some(anc.to_path_buf());
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join("vinkona");
        if looks_like_checkout(&p) {
            return Some(p);
        }
    }
    None
}

fn checkout(app: &AppHandle, stored: &State<Stored>) -> Result<PathBuf, String> {
    let cfg = stored.0.lock().unwrap().clone();
    if let Some(p) = find_checkout(&cfg.checkout) {
        // remember an auto-detected checkout so the next start is instant
        if cfg.checkout.as_deref() != Some(p.to_str().unwrap_or_default()) {
            let mut c = stored.0.lock().unwrap();
            c.checkout = Some(p.to_string_lossy().into_owned());
            save_cfg(app, &c);
        }
        return Ok(p);
    }
    Err("no Vinkona checkout found — pick its folder (the one holding vinkona.sh)".into())
}

#[tauri::command]
fn get_state(app: AppHandle, stored: State<Stored>) -> serde_json::Value {
    match checkout(&app, &stored) {
        Ok(p) => serde_json::json!({"checkout": p.to_string_lossy(), "valid": true}),
        Err(e) => serde_json::json!({"checkout": null, "valid": false, "error": e}),
    }
}

#[tauri::command]
fn pick_checkout(app: AppHandle, stored: State<Stored>) -> Result<String, String> {
    let picked = app
        .dialog()
        .file()
        .set_title("Where is Vinkona? (the folder holding vinkona.sh)")
        .blocking_pick_folder()
        .ok_or("cancelled")?;
    let p = picked.into_path().map_err(|e| e.to_string())?;
    if !looks_like_checkout(&p) {
        return Err(format!("{} doesn't look like a Vinkona checkout", p.display()));
    }
    let mut c = stored.0.lock().unwrap();
    c.checkout = Some(p.to_string_lossy().into_owned());
    save_cfg(&app, &c);
    Ok(p.to_string_lossy().into_owned())
}

#[tauri::command]
fn status(app: AppHandle, stored: State<Stored>) -> Result<serde_json::Value, String> {
    let dir = checkout(&app, &stored)?;
    let out = Command::new("python3")
        .arg(dir.join("assistant").join("supervisor.py"))
        .arg("status")
        .arg("--json")
        .current_dir(&dir)
        .output()
        .map_err(|e| format!("python3 failed: {e}"))?;
    serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("bad status JSON: {e}: {}", String::from_utf8_lossy(&out.stdout)))
}

#[tauri::command]
fn action(app: AppHandle, stored: State<Stored>, verb: String) -> Result<String, String> {
    if !matches!(verb.as_str(), "start" | "stop" | "restart") {
        return Err(format!("unknown action: {verb}"));
    }
    let dir = checkout(&app, &stored)?;
    let out = Command::new("bash")
        .arg("vinkona.sh")
        .arg(&verb)
        .current_dir(&dir)
        .output()
        .map_err(|e| format!("could not run vinkona.sh: {e}"))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if out.status.success() {
        Ok(text)
    } else {
        Err(text)
    }
}

#[tauri::command]
fn supervisor_json(
    app: AppHandle,
    stored: State<Stored>,
    args: Vec<String>,
) -> Result<serde_json::Value, String> {
    // The wizards' single pipe into the checkout's Python.  Whitelisted verbs
    // only — every one prints a single JSON object and owns its own safety
    // (config-patch validates, logtail refuses traversal, fetch-models
    // detaches).  `status --json` keeps its dedicated command above.
    let ok_verb = matches!(
        args.first().map(String::as_str),
        Some("preflight" | "vinur-probe" | "config-patch" | "fetch-models" | "logtail")
    );
    if !ok_verb {
        return Err(format!("verb not allowed: {:?}", args.first()));
    }
    let dir = checkout(&app, &stored)?;
    let out = Command::new("python3")
        .arg(dir.join("assistant").join("supervisor.py"))
        .args(&args)
        .current_dir(&dir)
        .output()
        .map_err(|e| format!("python3 failed: {e}"))?;
    serde_json::from_slice(&out.stdout).map_err(|e| {
        format!(
            "bad JSON from supervisor {:?}: {e}: {}",
            args.first(),
            String::from_utf8_lossy(&out.stdout)
        )
    })
}

#[tauri::command]
fn open_ui(app: AppHandle, which: String, url: String) -> Result<(), String> {
    // Only ever open the stack's own UIs: localhost, or the configured
    // remote knowledge host the status payload handed us.
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("not a web URL".into());
    }
    let label = match which.as_str() {
        "config" => "config-ui",
        "kb" => "kb-panel",
        _ => return Err(format!("unknown UI: {which}")),
    };
    if let Some(w) = app.get_webview_window(label) {
        let _ = w.set_focus();
        return Ok(());
    }
    let parsed: tauri::Url = url.parse().map_err(|e| format!("bad URL: {e}"))?;
    WebviewWindowBuilder::new(&app, label, WebviewUrl::External(parsed))
        .title(if label == "config-ui" { "Vinkona — Settings" } else { "Vinkona — Knowledge" })
        .inner_size(1150.0, 820.0)
        .build()
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    // WebKitGTK's DMABUF renderer crashes on NVIDIA's proprietary driver
    // (Wayland: instant "Error 71 (Protocol error)"; X11: blank window).
    // The ./Vinkona wrapper sets this too, but the binary must also survive
    // being launched directly.  Respect an explicit user choice.
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none()
            && Path::new("/proc/driver/nvidia/version").exists()
        {
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            get_state,
            pick_checkout,
            status,
            action,
            open_ui,
            supervisor_json
        ])
        .setup(|app| {
            let cfg = load_cfg(app.handle());
            app.manage(Stored(Mutex::new(cfg)));

            // Tray: the everyday surface — glance + start/stop without a window.
            let open = MenuItem::with_id(app, "open", "Open Vinkona", true, None::<&str>)?;
            let start = MenuItem::with_id(app, "start", "Start the stack", true, None::<&str>)?;
            let stop = MenuItem::with_id(app, "stop", "Stop the stack", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit launcher", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &start, &stop, &quit])?;
            TrayIconBuilder::with_id("main")
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .tooltip("Vinkona")
                .on_menu_event(|app, event| {
                    let id = event.id().as_ref().to_string();
                    match id.as_str() {
                        "open" => {
                            if let Some(w) = app.get_webview_window("main") {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                        "start" | "stop" => {
                            let app = app.clone();
                            std::thread::spawn(move || {
                                let stored: State<Stored> = app.state();
                                let _ = action(app.clone(), stored, id);
                            });
                        }
                        "quit" => app.exit(0),
                        _ => {}
                    }
                })
                .build(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running the Vinkona launcher");
}
