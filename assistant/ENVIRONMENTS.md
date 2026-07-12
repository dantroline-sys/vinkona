# Python environments

The default stack needs just ONE virtualenv: the default TTS engine
(`orpheus_gguf`) runs the Orpheus backbone on llama-server and vocodes with
onnxruntime inside `vinkona_env`, so no torch/vLLM ever gets installed. The
legacy engines keep their own venvs because their torch/vLLM pins conflict
with the core stack (and with each other).

| venv          | runs in          | what lives in it |
|---------------|------------------|------------------|
| `vinkona_env`   | host + container | The core Python services: the **cascade** (voice loop), **ASR** (faster-whisper) + soxr + rnnoise, the **memory** system, the **research worker**, the **config** web UI — and the default **orpheus_gguf TTS** (onnxruntime, CPU SNAC vocoder; the 3B backbone is a llama-server, not a Python dep). *(Formerly `personaplex_env` — renamed; Vinkona is a local cascade now, not PersonaPlex.)* |
| `orpheus_env` | container        | **Orpheus** TTS on vLLM (legacy path; ~6 GB). Built on Python 3.10–3.13 — vLLM's dependency chain (numba) doesn't support newer interpreters yet, so `install_orpheus.sh` picks a suitable one (and offers to install `python3.13` if the system only has a newer python). Only needed if you set `tts.engine` to `orpheus`. |
| `neutts_env`  | container        | **NeuTTS** (alternative TTS engine). |

The llama.cpp LM launchers — `serve_fast_lm.sh`, `serve_big_lm.sh`, `serve_big_lm2.sh`,
`serve_embed.sh`, `serve_tts_lm.sh` — use the system `python3`: they only exec
`llama-server` and import the standard library, so they need no venv.

None of these venvs belong in git: they're large (multiple GB), machine-specific, and have
absolute paths baked in. All are gitignored and rebuilt by the install scripts.

## Migrating an existing install

Already have a `personaplex_env`? Rename it once (it fixes the venv's baked paths too):

```bash
./rename_env.sh        # personaplex_env -> vinkona_env
./vinkona.sh restart
```

Fresh installs create `vinkona_env` directly.
