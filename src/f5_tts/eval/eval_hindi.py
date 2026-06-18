"""
Usage:
    python prepare_hindi.py huggingFaceLoginToken /output/dataset/path [--pretrain] [--workers N]
"""

import os
import sys
import argparse
from f5_tts.api import F5TTS

sys.path.append(os.getcwd())

def evaluate_hindi(ckpt, config, vocab, ref_audio, output_file):
    tts_hindi = F5TTS(ckpt_file=ckpt, vocab_file=vocab,model=config)
    wav, _, _ = tts_hindi.infer(
        ref_file=ref_audio,
        ref_text="हाँ हाँ सर डिपेंड आप पर करता है सर आप कितना टाइम लेते हैं क्योंकि लोकेशन और स्टेप तो आपको ही डिसाइड करना है।",
        gen_text="सर आप कितना टाइम लेते हैं क्योंकि लोकेशन और स्टेप तो आपको ही डिसाइड करना है।",
        file_wave=output_file,
        seed=None,
    )
    return wav

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint.")
    parser.add_argument("--ckpt", type=str, help="Model checkpoint file")
    parser.add_argument("--config-name", type=str, help="Configuration used to train the model")
    parser.add_argument("--vocab", type=str, help="Vocab file used to train the model")
    parser.add_argument("--ref-audio-file", type=str, help="Reference audio for generating matching speech")
    parser.add_argument("--output-file", type=str, help="Output audio file where generated speech is saved")
    return parser.parse_args()


def cli():
    try:
        args = get_args()
        evaluate_hindi(args.ckpt, args.config_name, args.vocab, args.ref_audio_file, args.output_file)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    cli()
