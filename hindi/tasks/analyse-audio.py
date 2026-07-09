"""
Analyse audio samples for leading and trailing pauses in IndicVoices-R Hindi.

For each sample the script detects where speech begins and ends using energy-based
voice activity detection (librosa.effects.trim), then records the durations of the
silence before speech starts (leading pause) and after speech ends (trailing pause).

Outputs saved to --out-dir:
  pause_dist.png               Side-by-side histogram of leading & trailing pauses
  leading_vs_trailing.png      Scatter: leading pause vs trailing pause
  pause_stats.json             Summary statistics (mean, std, percentiles)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import librosa


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def detect_pauses(audio_path, top_db):
    """Return (leading_pause_s, trailing_pause_s, total_duration_s)."""
    
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    total = len(wav) / sr
    _, (speech_start, speech_end) = librosa.effects.trim(wav, top_db=top_db)

    leading = speech_start / sr
    trailing = (len(wav) - speech_end) / sr
    return leading, trailing, total


def _stats(arr):
    """Return a dict of summary statistics (values in milliseconds)."""
    a = np.array(arr) * 1000  # convert s -> ms
    return {
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "p25_ms": float(np.percentile(a, 25)),
        "p50_ms": float(np.percentile(a, 50)),
        "p75_ms": float(np.percentile(a, 75)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "pct_under_100ms": float(100.0 * np.mean(a < 100)),
    }


def analyse_pauses(rows, out_dir, limit, top_db):
    n_rows = min(limit, len(rows)) if limit else len(rows)
    print(f"[pauses] Analysing {n_rows} samples  (top_db={top_db} dB)...")

    leading_pauses, trailing_pauses, total_durations = [], [], []
    errors = 0

    for row in tqdm(rows[:n_rows], desc="Detecting pauses"):
        try:
            lp, tp, dur = detect_pauses(row["audio_path"], top_db)
            leading_pauses.append(lp)
            trailing_pauses.append(tp)
            total_durations.append(dur)
        except Exception as e:
            print(f"  Warning: {row['id']}: {e}")
            errors += 1

    n_ok = len(leading_pauses)
    if n_ok == 0:
        print("[pauses] No samples processed successfully.")
        return

    stats = {
        "n_samples": n_ok,
        "n_errors": errors,
        "top_db_threshold": top_db,
        "leading_pause": _stats(leading_pauses),
        "trailing_pause": _stats(trailing_pauses),
    }

    with open(out_dir / "pause_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Clip x-axis to 99th percentile for readability
    all_vals_ms = (np.array(leading_pauses + trailing_pauses) * 1000)
    x_max = float(np.percentile(all_vals_ms, 99))
    bins = np.linspace(0, x_max, 60)

    # --- Plot 1: side-by-side histograms ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    lp_ms = np.array(leading_pauses) * 1000
    tp_ms = np.array(trailing_pauses) * 1000

    ax1.hist(lp_ms, bins=bins, edgecolor="black")
    ax1.axvline(stats["leading_pause"]["mean_ms"], color="red", linestyle="--",
                label=f"mean={stats['leading_pause']['mean_ms']:.0f}ms")
    ax1.axvline(stats["leading_pause"]["p50_ms"], color="orange", linestyle="--",
                label=f"p50={stats['leading_pause']['p50_ms']:.0f}ms")
    ax1.set_xlabel("Leading Pause (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Leading Pause Distribution")
    ax1.legend()

    ax2.hist(tp_ms, bins=bins, edgecolor="black")
    ax2.axvline(stats["trailing_pause"]["mean_ms"], color="red", linestyle="--",
                label=f"mean={stats['trailing_pause']['mean_ms']:.0f}ms")
    ax2.axvline(stats["trailing_pause"]["p50_ms"], color="orange", linestyle="--",
                label=f"p50={stats['trailing_pause']['p50_ms']:.0f}ms")
    ax2.set_xlabel("Trailing Pause (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("Trailing Pause Distribution")
    ax2.legend()

    fig.suptitle(f"Audio Pause Analysis  (n={n_ok}, top_db={top_db} dB)")
    plt.tight_layout()
    fig.savefig(out_dir / "pause_dist.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: scatter leading vs trailing ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(lp_ms, tp_ms, alpha=0.25, s=5)
    ax.set_xlabel("Leading Pause (ms)")
    ax.set_ylabel("Trailing Pause (ms)")
    ax.set_title(f"Leading vs Trailing Pause  (n={n_ok})")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, x_max)
    plt.tight_layout()
    fig.savefig(out_dir / "leading_vs_trailing.png", dpi=150)
    plt.close(fig)

    lp_s = stats["leading_pause"]
    tp_s = stats["trailing_pause"]
    print(
        f"[leading]  mean={lp_s['mean_ms']:.0f}ms  p50={lp_s['p50_ms']:.0f}ms  "
        f"p95={lp_s['p95_ms']:.0f}ms  under_100ms={lp_s['pct_under_100ms']:.1f}%"
    )
    print(
        f"[trailing] mean={tp_s['mean_ms']:.0f}ms  p50={tp_s['p50_ms']:.0f}ms  "
        f"p95={tp_s['p95_ms']:.0f}ms  under_100ms={tp_s['pct_under_100ms']:.1f}%"
    )
    print(f"[pauses] {errors} errors; outputs saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Analyse audio pauses (leading & trailing silence) in IndicVoices-R Hindi."
    )
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest with audio_path fields.")
    parser.add_argument("--out-dir", required=True, help="Directory for output plots and JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples (smoke test).")
    parser.add_argument(
        "--top-db", type=float, default=40.0,
        help="Energy threshold in dB below peak for silence detection (default: 40)."
    )
    return parser.parse_args()


def cli():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    print(f"Loaded {len(rows)} rows from {args.manifest}")

    analyse_pauses(rows, out_dir, args.limit, args.top_db)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    cli()
