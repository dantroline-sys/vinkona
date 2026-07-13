# Python environments

The default stack needs just ONE virtualenv: the default TTS engine
(`orpheus_gguf`) runs the Orpheus backbone on llama-server and vocodes with
onnxruntime inside `vinkona_env`, so no torch ever gets installed. The
alternative `neutts` engine needs torch, so it keeps its own venv.

Both are built by **uv** (bootstrapped in-tree by the install scripts — see
`env.sh` `vk_uv`), each from its own project so their resolutions never mix:
[`pyproject.toml`](pyproject.toml) + `uv.lock` pin `vinkona_env`, and
[`deps/neutts/`](deps/neutts/pyproject.toml) pins `neutts_env` — NeuTTS's
torch/numba stack caps numpy and lags new CPython releases, and a joint
resolution would drag the core venv down to its ceilings. One lockfile each
covers Linux, macOS, and Windows. If the system Python doesn't satisfy a
project's `requires-python`, uv downloads a matching CPython into `var/uv/`
instead of failing (this is how neutts_env gets 3.13 on a 3.14 system). The
venvs it produces are plain venvs; every script that calls
`vinkona_env/bin/python` works unchanged.

| venv          | runs in          | what lives in it |
|---------------|------------------|------------------|
| `vinkona_env`   | host + container | The core Python services: the **cascade** (voice loop), **ASR** (faster-whisper) + soxr + rnnoise, the **memory** system, the **research worker**, the **config** web UI — and the default **orpheus_gguf TTS** (onnxruntime, CPU SNAC vocoder; the 3B backbone is a llama-server, not a Python dep). |
| `neutts_env`  | container        | **NeuTTS** (alternative TTS engine — voice cloning; carries its own torch). |

The llama.cpp LM launchers — `serve_fast_lm.sh`, `serve_big_lm.sh`, `serve_big_lm2.sh`,
`serve_embed.sh`, `serve_tts_lm.sh` — use the system `python3`: they only exec
`llama-server` and import the standard library, so they need no venv.

None of these venvs belong in git: they're large (multiple GB), machine-specific, and have
absolute paths baked in. All are gitignored and rebuilt by the install scripts.
To change a dependency: edit `pyproject.toml`, run `uv lock` (the in-tree
binary lives at `bin/uv` after any install), commit both files.

## Migrating an existing install

Retired venvs from earlier iterations (`orpheus_env` from the vLLM TTS engine,
`personaplex_env` from before the cascade) are dead weight — `./install.sh
uninstall` removes them, or just `rm -rf` the directories.

Fresh installs create `vinkona_env` directly.
