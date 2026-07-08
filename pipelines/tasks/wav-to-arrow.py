import argparse
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
import soundfile as sf
import torch
import torchaudio

from f5_tts.model.modules import MelSpec
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

sys.path.append(os.getcwd())

TARGET_SAMPLE_RATE = 24000  # must match Vocos vocoder (execution plan)
N_MEL_CHANNELS = 100
HOP_LENGTH = 256

def to_mel_spec(row, mel_spec, resamplers):
    try:
        wav, sr = sf.read(row["audio_path"], dtype="float32")
        audio = torch.from_numpy(wav.T if getattr(wav, "ndim", 1) > 1 else wav).float()
        if sr != TARGET_SAMPLE_RATE:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
            audio = resamplers[sr](audio)
        audio = audio.reshape([1, -1])
        mel = mel_spec(audio).squeeze(0)
        return {"mel_spec": mel, "text": row["text"], "id": row["id"]}
    except Exception as e:
        print(f"Warning: mel extraction failed for {row['id']}: {e}. Skipping.")
        return None


def write_arrow_partition(shard_path, rows):
    with ArrowWriter(path=shard_path.as_posix()) as writer:
        for line in tqdm(rows, desc=f"Writing {shard_path.name}"):
            writer.write(line)
        writer.finalize()


def create_empty_arrow_partition(shard_path):
    with ArrowWriter(path=shard_path.as_posix()) as writer:
        writer.finalize()

def partition_to_rows(partition_index, rows_iter):
    mel_spec = MelSpec(
        target_sample_rate=TARGET_SAMPLE_RATE,
        n_mel_channels=N_MEL_CHANNELS,
        hop_length=HOP_LENGTH,
    )
    resamplers = {}

    for row in rows_iter:
        expanded_row = to_mel_spec(row, mel_spec, resamplers)
        if expanded_row is None:
            continue
        yield partition_index, expanded_row

def to_arrow(df, out_dir):
    num_partitions = df.rdd.getNumPartitions()
    total_rows = 0
    seen_partitions = set()
    current_partition = None
    current_writer = None

    try:
        partition_rows = df.rdd.mapPartitionsWithIndex(partition_to_rows).toLocalIterator()
        for partition_index, row in partition_rows:
            if partition_index != current_partition:
                if current_writer is not None:
                    current_writer.finalize()

                shard_path = out_dir / f"mel_{partition_index}.arrow"
                current_writer = ArrowWriter(path=shard_path.as_posix())
                current_partition = partition_index
                seen_partitions.add(partition_index)

            current_writer.write(row)
            total_rows += 1
    finally:
        if current_writer is not None:
            current_writer.finalize()

    for partition_index in range(num_partitions):
        if partition_index in seen_partitions:
            continue
        shard_path = out_dir / f"mel_{partition_index}.arrow"
        create_empty_arrow_partition(shard_path)

    print(f"\nFor {out_dir.stem}: {total_rows} samples")

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

    print(f"\nArrow files saved to: {args.out_dir} directory")
    spark.stop()

if __name__ == "__main__":
    cli()
