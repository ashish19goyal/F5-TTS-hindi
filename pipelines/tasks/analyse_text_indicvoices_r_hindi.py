"""
Analyse text in the IndicVoices-R Hindi JSONL manifest.

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
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm

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

def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_duration(row):
    if "duration" in row:
        return float(row["duration"])
    wav, sr = sf.read(row["audio_path"], dtype="float32")
    return len(wav) / sr


# ---------------------------------------------------------------------------
# 1.1 Character frequency
# ---------------------------------------------------------------------------

def analyse_char_freq(rows, out_dir):
    counter = Counter()
    for row in rows:
        counter.update(row["text"])

    with open(out_dir / "char_freq.json", "w", encoding="utf-8") as f:
        json.dump({"char_freq": dict(counter.most_common())}, f, ensure_ascii=False, indent=2)

    chars, counts = zip(*counter.most_common()) if counter else ([], [])
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

    print(f"[char_freq] {len(counter)} unique characters; top 5: {dict(counter.most_common(5))}")


# ---------------------------------------------------------------------------
# 1.2 Consonant-vowel combination frequency
# ---------------------------------------------------------------------------

def analyse_cv_combinations(rows, out_dir):
    counter = Counter()
    for row in rows:
        text = row["text"]
        for i in range(len(text) - 1):
            if text[i] in CONSONANTS and text[i + 1] in VOWEL_MARKS:
                counter[text[i] + text[i + 1]] += 1

    with open(out_dir / "cv_combo_freq.json", "w", encoding="utf-8") as f:
        json.dump({"cv_combo_freq": dict(counter.most_common())}, f, ensure_ascii=False, indent=2)

    top_n = min(60, len(counter))
    if not counter:
        print("[cv_combo] No consonant-vowel combinations found.")
        return

    combos, counts = zip(*counter.most_common(top_n))

    fig, ax = plt.subplots(figsize=(22, 6))
    ax.bar(range(top_n), counts)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(list(combos), fontsize=9)
    ax.set_xlabel("Consonant + Vowel Mark")
    ax.set_ylabel("Count")
    ax.set_title(f"Top {top_n} Consonant-Vowel Combination Frequencies  ({len(counter)} unique total)")
    plt.tight_layout()
    fig.savefig(out_dir / "cv_combo_freq.png", dpi=150)
    plt.close(fig)

    print(f"[cv_combo] {len(counter)} unique C+V combos; top 5: {dict(counter.most_common(5))}")


# ---------------------------------------------------------------------------
# 1.3 Duration distribution
# ---------------------------------------------------------------------------

def analyse_duration_dist(rows, out_dir, limit):
    n_rows = min(limit, len(rows)) if limit else len(rows)
    print(f"[duration] Computing durations for {n_rows} samples...")

    durations = []
    for row in tqdm(rows[:n_rows], desc="Reading durations"):
        try:
            durations.append(get_duration(row))
        except Exception as e:
            print(f"  Warning: {row['id']}: {e}")

    arr = np.array(durations)
    stats = {
        "count": int(len(arr)),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "p25_s": float(np.percentile(arr, 25)),
        "p50_s": float(np.percentile(arr, 50)),
        "p75_s": float(np.percentile(arr, 75)),
        "p95_s": float(np.percentile(arr, 95)),
        "p99_s": float(np.percentile(arr, 99)),
        "total_hours": float(arr.sum() / 3600),
    }

    with open(out_dir / "duration_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

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
    return durations


# ---------------------------------------------------------------------------
# 1.4 Correlation: text length vs audio duration
# ---------------------------------------------------------------------------

def analyse_text_audio_correlation(rows, durations, out_dir):
    n = len(durations)
    text_lengths = [len(row["text"]) for row in rows[:n]]

    corr = float(np.corrcoef(text_lengths, durations)[0, 1])

    with open(out_dir / "correlation.json", "w") as f:
        json.dump({"pearson_r": corr, "n_samples": n}, f, indent=2)

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
    return parser.parse_args()


def cli():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    print(f"Loaded {len(rows)} rows from {args.manifest}")

    analyse_char_freq(rows, out_dir)
    analyse_cv_combinations(rows, out_dir)
    durations = analyse_duration_dist(rows, out_dir, args.limit)
    analyse_text_audio_correlation(rows, durations, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    cli()
