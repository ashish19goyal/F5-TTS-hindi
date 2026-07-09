import argparse
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

sys.path.append(os.getcwd())

def build_vocab(df, out_dir):
    rows = (
        df.select(F.explode(F.split(F.col("text"), "")).alias("char"))
        .unique()
        .collect()
    )

    vocab_path = Path(out_dir) / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for ch in sorted(rows):
            f.write(ch + "\n")
    print(f"Vocab size: {len(rows)} -> {vocab_path}")

def get_args():
    parser = argparse.ArgumentParser(description="Build vocabulary for IndicVoices-R Hindi dataset.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest with text fields.")
    parser.add_argument("--out-dir", required=True, help="Folder for saving the generated vocab file (vocab.txt).")
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

    build_vocab(df, out_dir)

    print(f"\nVocab saved to: {out_dir}/vocab.txt")
    spark.stop()

if __name__ == "__main__":
    cli()
