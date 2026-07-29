"""
Inference pipeline for F5-TTS Hindi.

Environment variables (all optional):
  MODEL          Model name (default: F5TTS_Hindi)
  REF_AUDIO      Reference audio file path
  REF_TEXT       Reference audio transcript
  GEN_TEXT       Text to synthesize
  GEN_FILE       File with text to synthesize (overrides GEN_TEXT)
  OUTPUT_DIR     Output directory (default: tests)
  OUTPUT_FILE    Output wav filename (default: infer_out.wav)
  DEVICE         Device override
  VOCAB_FILE     Custom vocab file path
  CKPT_FILE      Custom checkpoint file path
  HF_CACHE_DIR   HuggingFace cache directory
  SEED           Random seed
"""

import codecs
import os
import random
import sys
from pathlib import Path

from importlib.resources import files as pkg_files


def main():
    model = os.environ.get("MODEL", "F5TTS_Hindi")
    ref_audio = os.environ.get("REF_AUDIO", "")
    ref_text = os.environ.get("REF_TEXT", "")
    gen_text = os.environ.get("GEN_TEXT", "")
    gen_file = os.environ.get("GEN_FILE", "")
    output_dir = os.environ.get("OUTPUT_DIR", "tests")
    output_file = os.environ.get("OUTPUT_FILE", "infer_out.wav")
    device = os.environ.get("DEVICE")
    vocab_file = os.environ.get("VOCAB_FILE", "")
    ckpt_file = os.environ.get("CKPT_FILE", "")
    hf_cache_dir = os.environ.get("HF_CACHE_DIR")
    seed = os.environ.get("SEED")

    # Default reference audio if not provided
    if not ref_audio:
        ref_audio = str(pkg_files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    if not ref_text:
        ref_text = "Some call me nature, others call me mother nature."

    # Read gen text from file if provided
    if gen_file:
        gen_text = codecs.open(gen_file, "r", "utf-8").read()

    if not gen_text:
        print("ERROR: Provide GEN_TEXT or GEN_FILE (environment variables).")
        sys.exit(1)

    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    wave_path = Path(output_dir) / output_file

    # Build kwargs for F5TTS
    f5_kwargs = {"model": model}
    if ckpt_file:
        f5_kwargs["ckpt_file"] = ckpt_file
    if vocab_file:
        f5_kwargs["vocab_file"] = vocab_file
    if device:
        f5_kwargs["device"] = device
    if hf_cache_dir:
        f5_kwargs["hf_cache_dir"] = hf_cache_dir

    from f5_tts.api import F5TTS

    f5tts = F5TTS(**f5_kwargs)

    if seed is not None:
        seed = int(seed)
    else:
        seed = random.randint(0, sys.maxsize)

    print(f"Model: {model}")
    print(f"Output: {wave_path}")
    print(f"Seed: {seed}")

    wav, sr, spec = f5tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=str(wave_path),
        seed=seed,
    )

    print(f"Generated {len(wav) / sr:.2f}s audio at {wave_path}")


if __name__ == "__main__":
    main()