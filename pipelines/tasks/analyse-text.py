"""
Analyse text in the IndicVoices-R Hindi JSONL manifest.

Uses PySpark to partition manifest rows across the cluster; plots and
final aggregations run on the driver after collecting lightweight results.

Analyses:
  1  Character frequency distribution across all utterances
  2  Consonant-vowel (C+matra) combination frequency distribution
  3  Audio duration distribution (uses manifest 'duration' field or reads audio)
  4  Correlation between audio duration and text length

Outputs saved to --out-dir:
  char_freq.png / char_freq.json
  cv_combo_freq.png / cv_combo_freq.json
  duration_dist.png / duration_stats.json
  text_audio_corr.png / correlation.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType

# Devanagari consonants (U+0915–U+0939) plus nukta forms (U+0958–U+095F)
CONSONANTS = frozenset(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
    "क़ख़ग़ज़ड़ढ़फ़य़"
)

# Vowel marks (matras): ā i ī u ū ṛ e ai o au + anusvara + visarga + chandrabindu
VOWEL_MARKS = frozenset(
    "\u093E\u093F\u0940\u0941\u0942\u0943\u0944"  # ा ि ी ु ू ृ ॄ
    "\u0945\u0946\u0947\u0948\u0949\u094A\u094B\u094C"  # ॅ ॆ े ै ॉ ॊ ो ौ
    "\u0902\u0903\u0901"  # ं ः ँ
)

plt.rcParams["font.family"] = [
    "Noto Sans Devanagari", "Lohit Devanagari", "Arial Unicode MS", "DejaVu Sans"
]


# ---------------------------------------------------------------------------
# Spark UDFs (module-level so they are picklable and sent to workers)
# ---------------------------------------------------------------------------

@F.udf(returnType=ArrayType(StringType()))
def extract_cv_pairs(text):
    """Extract consonant+vowel_mark bigrams from a text string."""
    consonants = frozenset(
        "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
        "क़ख़ग़ज़ड़ढ़फ़य़"
    )
    vowel_marks = frozenset(
        "\u093E\u093F\u0940\u0941\u0942\u0943\u0944"
        "\u0945\u0946\u0947\u0948\u0949\u094A\u094B\u094C"
        "\u0902\u0903\u0901"
    )
    if not text:
        return []
    pairs = []
    for i in range(len(text) - 1):
        if text[i] in consonants and text[i + 1] in vowel_marks:
            pairs.append(text[i] + text[i + 1])
    return pairs

# ---------------------------------------------------------------------------
# 1.1 Character frequency
# ---------------------------------------------------------------------------

def analyse_char_freq(df, out_dir):
    rows = (
        df.select(F.explode(F.split(F.col("text"), "")).alias("char"))
        .groupBy("char")
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )

    char_freq = {r["char"]: r["count"] for r in rows}
    with open(out_dir / "char_freq.json", "w", encoding="utf-8") as f:
        json.dump({"char_freq": char_freq}, f, ensure_ascii=False, indent=2)

    chars = [r["char"] for r in rows]
    counts = [r["count"] for r in rows]
    n = min(len(chars), 80)

    fig, ax = plt.subplots(figsize=(22, 6))
    ax.bar(range(n), counts[:n])
    ax.set_xticks(range(n))
    ax.set_xticklabels(list(chars[:n]), fontsize=9)
    ax.set_xlabel("Character")
    ax.set_ylabel("Count")
    ax.set_title(f"Character Frequency Distribution  (top {n} of {len(chars)} unique)")
    plt.tight_layout()
    fig.savefig(out_dir / "char_freq.png", dpi=150)
    plt.close(fig)

    top5 = {r["char"]: r["count"] for r in rows[:5]}
    print(f"[char_freq] {len(chars)} unique characters; top 5: {top5}")


# ---------------------------------------------------------------------------
# 1.2 Consonant-vowel combination frequency
# ---------------------------------------------------------------------------

def analyse_cv_combinations(df, out_dir):
    rows = (
        df.select(F.explode(extract_cv_pairs(F.col("text"))).alias("cv"))
        .groupBy("cv")
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )

    cv_freq = {r["cv"]: r["count"] for r in rows}
    with open(out_dir / "cv_combo_freq.json", "w", encoding="utf-8") as f:
        json.dump({"cv_combo_freq": cv_freq}, f, ensure_ascii=False, indent=2)

    top_n = min(60, len(rows))
    if not rows:
        print("[cv_combo] No consonant-vowel combinations found.")
        return

    combos = [r["cv"] for r in rows[:top_n]]
    counts = [r["count"] for r in rows[:top_n]]

    fig, ax = plt.subplots(figsize=(22, 6))
    ax.bar(range(top_n), counts)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(list(combos), fontsize=9)
    ax.set_xlabel("Consonant + Vowel Mark")
    ax.set_ylabel("Count")
    ax.set_title(f"Top {top_n} Consonant-Vowel Combination Frequencies  ({len(rows)} unique total)")
    plt.tight_layout()
    fig.savefig(out_dir / "cv_combo_freq.png", dpi=150)
    plt.close(fig)

    top5 = {r["cv"]: r["count"] for r in rows[:5]}
    print(f"[cv_combo] {len(rows)} unique C+V combos; top 5: {top5}")


# ---------------------------------------------------------------------------
# 1.3 Duration distribution
# ---------------------------------------------------------------------------

def analyse_duration_dist(df, out_dir, limit):
    dur_df = df.limit(limit) if limit else df

    dur_df = dur_df.withColumn("duration_s", F.col("duration").cast(FloatType()))
    dur_df = dur_df.filter(F.col("duration_s").isNotNull()).cache()

    n_rows = dur_df.count()
    print(f"[duration] Computing durations for {n_rows} samples...")

    stats_row = dur_df.select(
        F.count("duration_s").alias("count"),
        F.mean("duration_s").alias("mean_s"),
        F.stddev("duration_s").alias("std_s"),
        F.min("duration_s").alias("min_s"),
        F.max("duration_s").alias("max_s"),
        F.percentile_approx("duration_s", 0.25).alias("p25_s"),
        F.percentile_approx("duration_s", 0.50).alias("p50_s"),
        F.percentile_approx("duration_s", 0.75).alias("p75_s"),
        F.percentile_approx("duration_s", 0.95).alias("p95_s"),
        F.percentile_approx("duration_s", 0.99).alias("p99_s"),
        (F.sum("duration_s") / 3600).alias("total_hours"),
    ).first()

    stats = {k: float(v) for k, v in stats_row.asDict().items()}
    stats["count"] = int(stats_row["count"])

    with open(out_dir / "duration_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    durations = [r["duration_s"] for r in dur_df.select("duration_s").collect()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=60, edgecolor="black")
    ax.axvline(stats["mean_s"], color="red", linestyle="--", label=f"mean = {stats['mean_s']:.1f}s")
    ax.axvline(stats["p50_s"], color="orange", linestyle="--", label=f"median = {stats['p50_s']:.1f}s")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Audio Duration Distribution  "
        f"(n={len(durations)}, total={stats['total_hours']:.1f}h)"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "duration_dist.png", dpi=150)
    plt.close(fig)

    print(
        f"[duration] n={len(durations)}, mean={stats['mean_s']:.2f}s, "
        f"median={stats['p50_s']:.2f}s, total={stats['total_hours']:.1f}h"
    )
    return dur_df


# ---------------------------------------------------------------------------
# 1.4 Correlation: text length vs audio duration
# ---------------------------------------------------------------------------

def analyse_text_audio_correlation(dur_df, out_dir):
    paired_df = dur_df.withColumn("text_len", F.length("text")).cache()

    corr = paired_df.stat.corr("text_len", "duration_s")
    n = paired_df.count()

    with open(out_dir / "correlation.json", "w") as f:
        json.dump({"pearson_r": corr, "n_samples": n}, f, indent=2)

    pairs = paired_df.select("text_len", "duration_s").collect()
    text_lengths = [r["text_len"] for r in pairs]
    durations = [r["duration_s"] for r in pairs]

    m, b = np.polyfit(text_lengths, durations, 1)
    x = np.array([min(text_lengths), max(text_lengths)])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(text_lengths, durations, alpha=0.25, s=5)
    ax.plot(x, m * x + b, color="red", linewidth=1.5, label=f"fit: y={m:.3f}x+{b:.2f}")
    ax.set_xlabel("Text Length (characters)")
    ax.set_ylabel("Audio Duration (s)")
    ax.set_title(f"Text Length vs Audio Duration  (r={corr:.3f}, n={n})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "text_audio_corr.png", dpi=150)
    plt.close(fig)

    print(f"[correlation] Pearson r={corr:.4f}  (text_len vs audio_duration, n={n})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Analyse text in IndicVoices-R Hindi JSONL manifest.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest file.")
    parser.add_argument("--out-dir", required=True, help="Directory for output plots and JSON files.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit rows for audio reads (smoke test). Text-only analyses always use all rows."
    )
    parser.add_argument(
        "--num-partitions", type=int, default=None,
        help="Number of Spark partitions. Defaults to Spark's heuristic based on input size."
    )
    return parser.parse_args()


def cli():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("IndicVoices-R Hindi Text Analysis")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.json(args.manifest)
    if args.num_partitions:
        df = df.repartition(args.num_partitions)

    total = df.count()
    print(f"Loaded {total} rows from {args.manifest}")

    analyse_char_freq(df, out_dir)
    analyse_cv_combinations(df, out_dir)
    dur_df = analyse_duration_dist(df, out_dir, args.limit)
    analyse_text_audio_correlation(dur_df, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    spark.stop()


if __name__ == "__main__":
    cli()
