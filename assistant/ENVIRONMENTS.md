# Python environments

Vinkona runs across three isolated virtualenvs. The TTS engines need torch/vLLM versions that
conflict with the core stack (and with each other), so they're kept strictly apart.

| venv          | runs in          | what lives in it |
|---------------|------------------|------------------|
| `vinkona_env`   | host + container | The core Python services: the **cascade** (voice loop), **ASR** (faster-whisper) + soxr + rnnoise, the **memory** system, the **research worker**, and the **config** web UI. *(Formerly `personaplex_env` — renamed; Vinkona is a local cascade now, not PersonaPlex.)* |
| `orpheus_env` | container        | **Orpheus** TTS (on vLLM). Built on Python 3.10–3.13 — vLLM's dependency chain (numba) doesn't support newer interpreters yet, so `install_orpheus.sh` picks a suitable one (and offers to install `python3.13` if the system only has a newer python). |
| `neutts_env`  | container        | **NeuTTS** (alternative TTS engine). |

The llama.cpp LM launchers — `serve_fast_lm.sh`, `serve_big_lm.sh`, `serve_big_lm2.sh`,
`serve_embed.sh` — use the system `python3`: they only exec `llama-server` and import the
standard library, so they need no venv.

None of these venvs belong in git: they're large (multiple GB), machine-specific, and have
absolute paths baked in. All are gitignored and rebuilt by the install scripts.

## Migrating an existing install

Already have a `personaplex_env`? Rename it once (it fixes the venv's baked paths too):

```bash
./rename_env.sh        # personaplex_env -> vinkona_env
./vinkona.sh restart
```

Fresh installs create `vinkona_env` directly.
