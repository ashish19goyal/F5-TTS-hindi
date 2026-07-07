import argparse
import json
import os
import sys
from pathlib import Path
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm

sys.path.append(os.getcwd())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_DATASET_ID = "ai4bharat/indicvoices_r"
HF_DATASET_CONFIG = "hindi"
HF_DATASET_SPLIT = "train"

ARROW_SHARD_SIZE = 10000  # samples per .arrow shard in the package stage
DEFAULT_NUM_PARTITIONS = 64

# ---------------------------------------------------------------------------
# Manifest helpers  (rows: {"id": str, "audio_path": str, "text": str, ...})
# ---------------------------------------------------------------------------

def audio_dir(work_dir):
    d = Path(work_dir) / "audio" / "raw"
    d.mkdir(exist_ok=True, parents=True)
    return d


def write_manifest(work_dir, rows):
    path = Path(work_dir) / "manifests" / "raw.jsonl"
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {path}")


def download(args):
    """Download IndicVoices-R (hindi) and dump WAVs + a JSONL manifest.

    Audio is materialized as files so that downstream Spark stages can address
    samples by path instead of shipping the HF dataset object to executors.
    """

    if args.hf_token:
        login(args.hf_token)

    dataset = load_dataset(HF_DATASET_ID, HF_DATASET_CONFIG, split=HF_DATASET_SPLIT)
    print(f"Downloaded {HF_DATASET_ID} ({HF_DATASET_CONFIG}/{HF_DATASET_SPLIT})")
    print(dataset)

    audio_column = "audio"
    text_column = "normalized" if "normalized" in dataset.column_names else "text"

    out_audio = audio_dir(args.work_dir)
    rows = []
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    for i in tqdm(range(total), desc="Exporting raw audio"):
        sample = dataset[i]
        audio = sample[audio_column]
        text = sample[text_column]
        try:
            wav_path = out_audio / f"{i:07d}.wav"
            sf.write(wav_path.as_posix(), audio["array"], audio["sampling_rate"], subtype="PCM_16")
            rows.append({"id": f"{i:07d}", "audio_path": wav_path.as_posix(), "text": text})
        except Exception as e:
            print(f"Warning: failed to export sample {i}: {e}. Skipping.")

    write_manifest(args.work_dir, rows)


def get_args():
    parser = argparse.ArgumentParser(description="Download IndicVoices-R Hindi dataset for F5-TTS.")
    parser.add_argument("--work-dir", required=True, help="Shared directory for intermediate audio + manifests.")
    parser.add_argument("--out-dir", help="Final dataset output directory (package stage).")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (download stage).")
    parser.add_argument("--spark-master", default=None, help='Spark master URL, e.g. "spark://host:7077" or "local[*]". Defaults to spark-submit config.')
    parser.add_argument("--num-partitions", type=int, default=DEFAULT_NUM_PARTITIONS, help="Spark partitions per stage.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N samples (smoke tests).")
    return parser.parse_args()


def cli():
    args = get_args()
    download(args)


if __name__ == "__main__":
    cli()
