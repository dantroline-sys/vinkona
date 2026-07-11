import argparse
from pathlib import Path
import mlx.core as mx
from personaplex_mlx.persona_utils import (
    get_lm_config,
    get_or_download_model_file,
    load_lm_weights,
    get_voice_prompt_dir,
    resolve_voice_prompt,
)
from personaplex_mlx import models
import personaplex_mlx.utils as utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", type=str, default="nvidia/personaplex-7b-v1")
    parser.add_argument("-q", "--quantized", type=int, default=8)
    parser.add_argument("--lm-config", type=str, default=None)
    parser.add_argument("--moshi-weight", type=str, default=None)
    parser.add_argument("--voice-prompt-dir", type=str, default=None)
    parser.add_argument("--voice", type=str, default="NATF0")
    args = parser.parse_args()

    print("🔄 Loading config...")
    lm_config = get_lm_config(args.lm_config, args.hf_repo)

    print("🔄 Resolving model file...")
    model_file, _ = get_or_download_model_file(
        hf_repo=args.hf_repo,
        quantized=args.quantized,
        explicit_model_file=args.moshi_weight,
    )

    print(f"🔄 Creating model (Q{args.quantized})...")
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    print("🔄 Loading weights...")
    load_lm_weights(model, lm_config, model_file, args.quantized)
    print("✅ Weights loaded successfully")

    print("🔄 Creating LmGen...")
    gen = models.LmGen(
        model=model,
        max_steps=2048,
        text_sampler=utils.Sampler(temp=0.8, top_k=250),
        audio_sampler=utils.Sampler(temp=0.8, top_k=250),
        check=False,
    )
    print("✅ LmGen created successfully")

    try:
        voice_prompt_dir = get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
        voice_prompt_path = resolve_voice_prompt(
            voice=args.voice, voice_prompt=None, voice_prompt_dir=voice_prompt_dir
        )
        gen.load_voice_prompt_embeddings(voice_prompt_path)
        print("✅ Voice prompt loaded")
    except Exception as e:
        print(f"⚠️  Skipping voice prompt: {e}")

    print("\n🔄 Running a few generation steps...")

    # Correct shape expected by LmGen.step(): (batch, codebooks, time)
    dummy_audio = mx.zeros((1, 8, 1))

    for i in range(8):
        text_token = gen.step(input_tokens=dummy_audio)
        audio_tokens = gen.last_audio_tokens()
        print(f"  Step {i+1}: text_token={text_token}, audio_tokens shape={audio_tokens.shape if audio_tokens is not None else None}")

    print("\n✅✅✅  MLX + PersonaPlex basic inference is working!")
    print(f"Active MLX memory: {mx.get_active_memory() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()

