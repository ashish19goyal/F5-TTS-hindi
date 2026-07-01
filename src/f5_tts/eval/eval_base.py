"""
Usage:
    python prepare_hindi.py huggingFaceLoginToken /output/dataset/path [--pretrain] [--workers N]
"""

import os
import sys
import argparse
from f5_tts.api import F5TTS

sys.path.append(os.getcwd())

def evaluate_base(config, vocab, ref_audio, output_file):
    tts_base = F5TTS(vocab_file=vocab,model=config)
    num_trainable_params = sum(p.numel() for p in tts_base.ema_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_trainable_params:,}")
    wav, _, _ = tts_base.infer(
        ref_file=ref_audio,
        ref_text="My name is Ashish. I live in Bengalore. I am 37 years old and I am working on a project to build text to speech AI platform.",
        gen_text="How many apples do you have?",
        file_wave=output_file,
        seed=None,
    )
    return wav

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint.")
    parser.add_argument("--config-name", type=str, help="Configuration used to train the model")
    parser.add_argument("--vocab", type=str, help="Vocab file used to train the model")
    parser.add_argument("--ref-audio-file", type=str, help="Reference audio for generating matching speech")
    parser.add_argument("--output-file", type=str, help="Output audio file where generated speech is saved")
    return parser.parse_args()


def cli():
    try:
        args = get_args()
        evaluate_base(args.config_name, args.vocab, args.ref_audio_file, args.output_file)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    cli()
