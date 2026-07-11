import time
import numpy as np
import mlx.core as mx
from personaplex_mlx import models, utils
from personaplex_mlx.persona_utils import (
    get_lm_config, get_or_download_model_file, load_lm_weights,
    get_or_download_mimi,
)
import rustymimi

HF = "nvidia/personaplex-7b-v1"
Q = 8
LM_CONFIG = "Models/personaplex-7b-v1-raw/config.json"
MOSHI = "Models/personaplex-7b-v1-raw/model.safetensors"

print("loading config + weights...")
cfg = get_lm_config(LM_CONFIG, HF)
model_file, _ = get_or_download_model_file(hf_repo=HF, quantized=Q, explicit_model_file=MOSHI)
model = models.Lm(cfg)
model.set_dtype(mx.bfloat16)
load_lm_weights(model, cfg, model_file, Q)
print("weights loaded")

gen = models.LmGen(
    model=model, max_steps=4096,
    text_sampler=utils.Sampler(temp=0.8, top_k=250),
    audio_sampler=utils.Sampler(temp=0.8, top_k=250),
    check=False, audio_silence_frame_cnt=int(0.5 * 12.5),
)
gen.reset_streaming()
gen.text_prompt_tokens = None
gen.step_system_prompts()

mimi_file = get_or_download_mimi(HF, None)
tok = rustymimi.StreamTokenizer(mimi_file, num_codebooks=8)
print(f"mimi: {mimi_file}")

# warmup mimi encoder
for _ in range(8):
    tok.encode(np.zeros(1920, dtype=np.float32))
    for _ in range(200):
        e = tok.get_encoded()
        if e is not None: break
        time.sleep(0.001)

print("\nbudget per frame = 80.0 ms (must be under this for real-time duplex)\n")
enc_t, step_t, dec_t = [], [], []
for i in range(40):
    pcm = (np.random.randn(1920).astype(np.float32) * 0.01)
    t0 = time.time()
    tok.encode(pcm)
    tokens = None
    for _ in range(200):
        tokens = tok.get_encoded()
        if tokens is not None: break
        time.sleep(0.001)
    t1 = time.time()
    if tokens is None:
        print(f"  step {i}: no encoded tokens"); continue
    mi = mx.array(tokens).transpose(1, 0)[:, : gen.user_codebooks][:, :, None]
    txt = gen.step(input_tokens=mi)
    mx.eval(txt)
    at = gen.last_audio_tokens()
    t2 = time.time()
    if at is not None:
        tok.decode(np.array(at).astype(np.uint32))
        while tok.get_decoded() is not None: pass
    t3 = time.time()
    if i >= 5:  # skip warmup
        enc_t.append((t1-t0)*1000); step_t.append((t2-t1)*1000); dec_t.append((t3-t2)*1000)
    total = (t3-t0)*1000
    flag = "  ✅" if total < 80 else "  ❌ OVER BUDGET"
    print(f"  step {i:2d}: encode={ (t1-t0)*1000:6.1f}  gen.step={(t2-t1)*1000:6.1f}  decode={(t3-t2)*1000:6.1f}  total={total:6.1f} ms{flag}")

import statistics as st
print("\n--- medians (steady state) ---")
print(f"  encode  : {st.median(enc_t):6.1f} ms")
print(f"  gen.step: {st.median(step_t):6.1f} ms")
print(f"  decode  : {st.median(dec_t):6.1f} ms")
tot = st.median(enc_t)+st.median(step_t)+st.median(dec_t)
print(f"  TOTAL   : {tot:6.1f} ms  (budget 80.0 ms)  -> {'REAL-TIME OK' if tot<80 else 'TOO SLOW for duplex'}")
print(f"  realtime factor: {tot/80.0:.2f}x  (1.0 = exactly real-time)")
