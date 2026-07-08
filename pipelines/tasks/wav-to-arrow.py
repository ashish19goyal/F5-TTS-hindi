import argparse
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
import soundfile as sf
import torch

from f5_tts.model.modules import MelSpec
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

sys.path.append(os.getcwd())

TARGET_SAMPLE_RATE = 24000  # must match Vocos vocoder (execution plan)
N_MEL_CHANNELS = 100
HOP_LENGTH = 256

def to_arrow(df, out_dir):
    
    def partition_fn(rows_iter):
    
        mel_spec = MelSpec(
            target_sample_rate=TARGET_SAMPLE_RATE, n_mel_channels=N_MEL_CHANNELS, hop_length=HOP_LENGTH
        )
        for row in rows_iter:
            try:
                wav, sr = sf.read(row["audio_path"], dtype="float32")
                if sr != TARGET_SAMPLE_RATE:
                    raise ValueError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {sr}; run standardize first.")
                audio = torch.tensor(wav).reshape([1, -1])
                mel = mel_spec(audio).squeeze(0)
                yield {"mel_spec": mel.numpy().tolist(), "text": row["text"], "duration": row["duration"]}
            except Exception as e:
                print(f"Warning: mel extraction failed for {row['id']}: {e}. Skipping.")

    rows = read_manifest(args.work_dir, previous_stage("package"))
    results = run_spark_stage("package", rows, partition_fn, args.spark_master, args.num_partitions)


    durations = []
    for shard_start in range(0, len(results), ARROW_SHARD_SIZE):
        shard = results[shard_start : shard_start + ARROW_SHARD_SIZE]
        shard_path = out_dir / f"mel_{shard_start}.arrow"
        with ArrowWriter(path=shard_path.as_posix()) as writer:
            for line in tqdm(shard, desc=f"Writing {shard_path.name}"):
                writer.write(line)
                durations.append(line["duration"])
            writer.finalize()

    with open((out_dir / "duration.json").as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)

    work_vocab = Path(args.work_dir) / "vocab.txt"
    (out_dir / "vocab.txt").write_text(work_vocab.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\nFor {out_dir.stem}: {len(results)} samples, {sum(durations) / 3600:.2f} hours")

def get_args():
    parser = argparse.ArgumentParser(description="Build arrow files from audio wavs for training F5TTS-hindi model.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest with text and audio file path fields.")
    parser.add_argument("--out-dir", required=True, help="Folder for saving the generated arrow files (mel_*.arrow)")
    return parser.parse_args()
 
def cli():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("IndicVoices-R Vocab builder")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.json(args.manifest)

    to_arrow(df, out_dir)

    print(f"\nArrow files saved to: {out_dir} directory")
    spark.stop()

if __name__ == "__main__":
    cli()
