# Python environments

The default stack needs just ONE virtualenv: the default TTS engine
(`orpheus_gguf`) runs the Orpheus backbone on llama-server and vocodes with
onnxruntime inside `vinkona_env`, so no torch ever gets installed. The
alternative `neutts` engine needs torch, so it keeps its own venv.

| venv          | runs in          | what lives in it |
|---------------|------------------|------------------|
| `vinkona_env`   | host + container | The core Python services: the **cascade** (voice loop), **ASR** (faster-whisper) + soxr + rnnoise, the **memory** system, the **research worker**, the **config** web UI — and the default **orpheus_gguf TTS** (onnxruntime, CPU SNAC vocoder; the 3B backbone is a llama-server, not a Python dep). |
| `neutts_env`  | container        | **NeuTTS** (alternative TTS engine — voice cloning; carries its own torch). |

The llama.cpp LM launchers — `serve_fast_lm.sh`, `serve_big_lm.sh`, `serve_big_lm2.sh`,
`serve_embed.sh`, `serve_tts_lm.sh` — use the system `python3`: they only exec
`llama-server` and import the standard library, so they need no venv.

None of these venvs belong in git: they're large (multiple GB), machine-specific, and have
absolute paths baked in. All are gitignored and rebuilt by the install scripts.

## Migrating an existing install

Retired venvs from earlier iterations (`orpheus_env` from the vLLM TTS engine,
`personaplex_env` from before the cascade) are dead weight — `./install.sh
uninstall` removes them, or just `rm -rf` the directories.

Fresh installs create `vinkona_env` directly.
