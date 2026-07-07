"""
Prepare the IndicVoices-R Hindi dataset for F5-TTS training.

The pipeline follows docs/execution-plan.md and is split into independent stages so
that an orchestrator (Airflow, see dags/prepare_indicvoices_r_dag.py) can run and
retry each stage separately. Every audio stage is distributed with PySpark: the
manifest rows are parallelized across executors and each partition lazily loads the
heavy model once (HTDemucs / VoiceFixer / DeepFilterNet3 / quality estimators).

NOTE: executors must share the filesystem holding --work-dir (NFS/Lustre/HDFS-fuse)
since stages exchange data as WAV files + JSONL manifests.

Stages (in order):
    download        Download ai4bharat/indicvoices_r (hindi) from Hugging Face,
                    dump WAVs + manifest.
    demix           Stage 1 - HTDemucs vocal stem extraction (background removal).
    dereverb        Stage 2 - VoiceFixer dereverberation / restoration.
    enhance         Stage 3 - DeepFilterNet3 residual noise suppression.
    quality_filter  Stage 4 - WADA-SNR, (optional) DNSMOS, speaking rate, pitch
                    variation, duration 3-20 s.
    standardize     Stage 5 - resample 24 kHz mono 16-bit, loudness normalize to
                    -21 LUFS, trim leading/trailing silence > 300 ms.
    text_process    Text pre-processing - NFC + IndicNLP normalization, numeral /
                    currency expansion, danda standardization, nukta consistency
                    (via NFC), drop utterances containing Latin script.
    build_vocab     Extract character inventory -> vocab.txt, assert zero OOV.
    package         Compute 24 kHz / 100-dim / hop-256 mel spectrograms and write
                    Arrow shards + duration.json + vocab.txt to --out-dir.

Usage (single stage, as Airflow runs it):
    python prepare_indicvoices_r.py download --hf-token TOKEN --work-dir /data/ivr
    spark-submit prepare_indicvoices_r.py demix --work-dir /data/ivr
    ...
    python prepare_indicvoices_r.py package --work-dir /data/ivr --out-dir /data/ivr_dataset

Usage (everything locally, for a smoke test):
    python prepare_indicvoices_r.py all --hf-token TOKEN --work-dir /data/ivr \
        --out-dir /data/ivr_dataset --spark-master "local[*]" --limit 50
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.append(os.getcwd())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_DATASET_ID = "ai4bharat/indicvoices_r"
HF_DATASET_CONFIG = "hindi"
HF_DATASET_SPLIT = "train"

TARGET_SAMPLE_RATE = 24000  # must match Vocos vocoder (execution plan)
N_MEL_CHANNELS = 100
HOP_LENGTH = 256

MIN_DURATION_S = 3.0
MAX_DURATION_S = 20.0
MIN_WADA_SNR_DB = 15.0
MIN_DNSMOS = 2.8  # bottom-tier perceptual quality cutoff
MIN_C50_DB = 40.0  # only enforced when a C50 estimator is available
SPEAKING_RATE_BOUNDS = (4.0, 30.0)  # Devanagari chars / second
MIN_PITCH_STD_HZ = 5.0  # drops flat / degenerate clips
TARGET_LUFS = -21.0  # middle of the -23..-20 LUFS band
TRIM_TOP_DB = 40
EDGE_SILENCE_S = 0.3  # keep at most 300 ms of edge silence

ARROW_SHARD_SIZE = 10000  # samples per .arrow shard in the package stage
DEFAULT_NUM_PARTITIONS = 64

STAGE_ORDER = [
    "download",
    "demix",
    "dereverb",
    "enhance",
    "quality_filter",
    "standardize",
    "text_process",
    "build_vocab",
    "package",
]

# ---------------------------------------------------------------------------
# Manifest helpers  (rows: {"id": str, "audio_path": str, "text": str, ...})
# ---------------------------------------------------------------------------


def manifest_path(work_dir, stage):
    return Path(work_dir) / "manifests" / f"{stage}.jsonl"


def stage_audio_dir(work_dir, stage):
    d = Path(work_dir) / "audio" / stage
    d.mkdir(exist_ok=True, parents=True)
    return d


def read_manifest(work_dir, stage):
    path = manifest_path(work_dir, stage)
    if not path.exists():
        raise FileNotFoundError(f"Manifest for stage '{stage}' not found at {path}. Run that stage first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(work_dir, stage, rows):
    path = manifest_path(work_dir, stage)
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{stage}] wrote {len(rows)} rows -> {path}")


def previous_stage(stage):
    return STAGE_ORDER[STAGE_ORDER.index(stage) - 1]


# ---------------------------------------------------------------------------
# Spark helpers
# ---------------------------------------------------------------------------


def get_spark(app_name, master=None):
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    return builder.getOrCreate()


def run_spark_stage(stage, rows, partition_fn, spark_master, num_partitions):
    """Parallelize manifest rows across the cluster and apply partition_fn per partition.

    partition_fn receives an iterator of rows and yields processed rows (or nothing
    for rows that failed / were filtered out). It must import its heavy deps and
    load models inside itself so the work happens on the executors.
    """
    spark = get_spark(f"ivr-{stage}", spark_master)
    rdd = spark.sparkContext.parallelize(rows, min(num_partitions, max(1, len(rows))))
    results = rdd.mapPartitions(partition_fn).collect()
    spark.stop()
    print(f"[{stage}] {len(results)}/{len(rows)} rows survived")
    return results


# ---------------------------------------------------------------------------
# Stage: download
# ---------------------------------------------------------------------------


def stage_download(args):
    """Download IndicVoices-R (hindi) and dump WAVs + a JSONL manifest.

    Audio is materialized as files so that downstream Spark stages can address
    samples by path instead of shipping the HF dataset object to executors.
    """
    import soundfile as sf
    from datasets import load_dataset
    from huggingface_hub import login
    from tqdm import tqdm

    if args.hf_token:
        login(args.hf_token)

    dataset = load_dataset(HF_DATASET_ID, HF_DATASET_CONFIG, split=HF_DATASET_SPLIT)
    print(f"Downloaded {HF_DATASET_ID} ({HF_DATASET_CONFIG}/{HF_DATASET_SPLIT})")
    print(dataset)

    audio_column = "audio_filepath" if "audio_filepath" in dataset.column_names else "audio"
    text_column = "text" if "text" in dataset.column_names else "normalized"

    out_audio = stage_audio_dir(args.work_dir, "download")
    rows = []
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    for i in tqdm(range(total), desc="Exporting raw audio"):
        sample = dataset[i]
        audio = sample[audio_column]
        text = sample[text_column]
        try:
            wav_path = out_audio / f"ivr_{i:07d}.wav"
            sf.write(wav_path.as_posix(), audio["array"], audio["sampling_rate"], subtype="PCM_16")
            rows.append({"id": f"ivr_{i:07d}", "audio_path": wav_path.as_posix(), "text": text})
        except Exception as e:
            print(f"Warning: failed to export sample {i}: {e}. Skipping.")

    write_manifest(args.work_dir, "download", rows)


# ---------------------------------------------------------------------------
# Stage 1: demix (HTDemucs)
# ---------------------------------------------------------------------------


def stage_demix(args):
    out_dir = stage_audio_dir(args.work_dir, "demix").as_posix()

    def partition_fn(rows_iter):
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model("htdemucs").to(device)
        model.eval()
        vocals_idx = model.sources.index("vocals")

        for row in rows_iter:
            try:
                wav, sr = torchaudio.load(row["audio_path"])
                if sr != model.samplerate:
                    wav = torchaudio.functional.resample(wav, sr, model.samplerate)
                if wav.shape[0] == 1:  # htdemucs expects stereo
                    wav = wav.repeat(2, 1)
                ref = wav.mean(0)
                wav = (wav - ref.mean()) / (ref.std() + 1e-8)
                with torch.no_grad():
                    sources = apply_model(model, wav[None].to(device), device=device)[0]
                vocals = sources[vocals_idx] * (ref.std() + 1e-8) + ref.mean()
                vocals = vocals.mean(0, keepdim=True).cpu()  # back to mono
                out_path = os.path.join(out_dir, os.path.basename(row["audio_path"]))
                torchaudio.save(out_path, vocals, model.samplerate)
                yield {**row, "audio_path": out_path}
            except Exception as e:
                print(f"Warning: demix failed for {row['id']}: {e}. Skipping.")

    rows = read_manifest(args.work_dir, previous_stage("demix"))
    results = run_spark_stage("demix", rows, partition_fn, args.spark_master, args.num_partitions)
    write_manifest(args.work_dir, "demix", results)


# ---------------------------------------------------------------------------
# Stage 2: dereverb (VoiceFixer)
# ---------------------------------------------------------------------------


def stage_dereverb(args):
    out_dir = stage_audio_dir(args.work_dir, "dereverb").as_posix()

    def partition_fn(rows_iter):
        import torch
        from voicefixer import VoiceFixer

        vf = VoiceFixer()
        use_cuda = torch.cuda.is_available()

        for row in rows_iter:
            try:
                out_path = os.path.join(out_dir, os.path.basename(row["audio_path"]))
                # mode 0: original restoration model (denoise + dereverb + declip)
                vf.restore(input=row["audio_path"], output=out_path, cuda=use_cuda, mode=0)
                yield {**row, "audio_path": out_path}
            except Exception as e:
                print(f"Warning: dereverb failed for {row['id']}: {e}. Skipping.")

    rows = read_manifest(args.work_dir, previous_stage("dereverb"))
    results = run_spark_stage("dereverb", rows, partition_fn, args.spark_master, args.num_partitions)
    write_manifest(args.work_dir, "dereverb", results)


# ---------------------------------------------------------------------------
# Stage 3: enhance (DeepFilterNet3)
# ---------------------------------------------------------------------------


def stage_enhance(args):
    out_dir = stage_audio_dir(args.work_dir, "enhance").as_posix()

    def partition_fn(rows_iter):
        import torchaudio
        from df.enhance import enhance, init_df, load_audio, save_audio

        model, df_state, _ = init_df()  # loads DeepFilterNet3 by default

        for row in rows_iter:
            try:
                audio, _ = load_audio(row["audio_path"], sr=df_state.sr())
                enhanced = enhance(model, df_state, audio)
                out_path = os.path.join(out_dir, os.path.basename(row["audio_path"]))
                save_audio(out_path, enhanced, df_state.sr())
                yield {**row, "audio_path": out_path}
            except Exception as e:
                print(f"Warning: enhance failed for {row['id']}: {e}. Skipping.")

    rows = read_manifest(args.work_dir, previous_stage("enhance"))
    results = run_spark_stage("enhance", rows, partition_fn, args.spark_master, args.num_partitions)
    write_manifest(args.work_dir, "enhance", results)


# ---------------------------------------------------------------------------
# Stage 4: quality_filter
# ---------------------------------------------------------------------------


def wada_snr(wav):
    """Blind SNR estimate (WADA-SNR, Kim & Stern 2008) from a float waveform."""
    import numpy as np

    # Precomputed table mapping gamma statistic -> SNR dB
    db_vals = np.arange(-20, 101)
    g_vals = np.array(
        [0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627,
         0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192 , 0.42077252, 0.42452799,
         0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
         0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724,
         0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418,
         0.95655449, 0.9835349 , 1.01047155, 1.0362095 , 1.06136425, 1.08579312, 1.1094819 , 1.13277995,
         1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029,
         1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727 , 1.37345513, 1.38577122, 1.39733504,
         1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
         1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097 , 1.528578  ,
         1.53389835, 1.5391211 , 1.5439065 , 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348,
         1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
         1.5941969 , 1.59693155, 1.599446  , 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547,
         1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119,
         1.62390423, 1.62513143, 1.62632463, 1.6274027 , 1.62842767, 1.62945532, 1.6303307 , 1.63128026,
         1.63204102]
    )

    wav = np.asarray(wav, dtype=np.float64)
    wav = wav - wav.mean()
    abs_wav = np.abs(wav)
    abs_wav[abs_wav < 1e-10] = 1e-10
    # gamma statistic: log of arithmetic mean minus mean of logs
    gamma = np.log(abs_wav.mean()) - np.log(abs_wav).mean()
    idx = np.searchsorted(g_vals, gamma)
    idx = min(max(idx, 0), len(db_vals) - 1)
    return float(db_vals[idx])


def estimate_c50(wav, sr):
    """C50 speech clarity estimate in dB, or None when no estimator is available.

    Blind C50 estimation needs a trained regressor (e.g. Brouhaha); it is not
    pip-installable in a portable way, so this hook returns None by default and
    the C50 check is skipped. Plug your estimator here to enforce MIN_C50_DB.
    """
    return None


def stage_quality_filter(args):
    def partition_fn(rows_iter):
        import librosa
        import numpy as np
        import soundfile as sf

        try:  # DNSMOS is optional; filter degrades gracefully without it
            from speechmos import dnsmos

            def mos_of(wav, sr):
                wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
                return float(dnsmos.run(wav16, sr=16000)["ovrl_mos"])
        except ImportError:
            print("Warning: 'speechmos' not installed; skipping DNSMOS filtering.")
            mos_of = None

        devanagari = re.compile(r"[ऀ-ॿ]")

        for row in rows_iter:
            try:
                wav, sr = sf.read(row["audio_path"], dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                duration = len(wav) / sr

                reasons = []
                if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
                    reasons.append(f"duration={duration:.1f}s")

                snr = wada_snr(wav)
                if snr < MIN_WADA_SNR_DB:
                    reasons.append(f"snr={snr:.1f}dB")

                c50 = estimate_c50(wav, sr)
                if c50 is not None and c50 < MIN_C50_DB:
                    reasons.append(f"c50={c50:.1f}dB")

                if mos_of is not None:
                    mos = mos_of(wav, sr)
                    if mos < MIN_DNSMOS:
                        reasons.append(f"dnsmos={mos:.2f}")

                n_chars = len(devanagari.findall(row["text"]))
                rate = n_chars / duration if duration > 0 else 0.0
                if not (SPEAKING_RATE_BOUNDS[0] <= rate <= SPEAKING_RATE_BOUNDS[1]):
                    reasons.append(f"speaking_rate={rate:.1f}chars/s")

                f0, _, _ = librosa.pyin(
                    wav, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6")
                )
                f0 = f0[~np.isnan(f0)]
                pitch_std = float(np.std(f0)) if len(f0) > 10 else 0.0
                if pitch_std < MIN_PITCH_STD_HZ:
                    reasons.append(f"pitch_std={pitch_std:.1f}Hz")

                if reasons:
                    print(f"Filtered {row['id']}: {', '.join(reasons)}")
                    continue
                yield {**row, "duration": duration, "snr": snr}
            except Exception as e:
                print(f"Warning: quality check failed for {row['id']}: {e}. Skipping.")

    rows = read_manifest(args.work_dir, previous_stage("quality_filter"))
    results = run_spark_stage("quality_filter", rows, partition_fn, args.spark_master, args.num_partitions)
    write_manifest(args.work_dir, "quality_filter", results)


# ---------------------------------------------------------------------------
# Stage 5: standardize
# ---------------------------------------------------------------------------


def stage_standardize(args):
    out_dir = stage_audio_dir(args.work_dir, "standardize").as_posix()

    def partition_fn(rows_iter):
        import librosa
        import numpy as np
        import pyloudnorm
        import soundfile as sf

        meter = pyloudnorm.Meter(TARGET_SAMPLE_RATE)

        for row in rows_iter:
            try:
                wav, sr = sf.read(row["audio_path"], dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != TARGET_SAMPLE_RATE:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

                # Trim edge silence but keep up to EDGE_SILENCE_S of padding
                _, (start, end) = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
                pad = int(EDGE_SILENCE_S * TARGET_SAMPLE_RATE)
                wav = wav[max(0, start - pad) : min(len(wav), end + pad)]

                loudness = meter.integrated_loudness(wav)
                wav = pyloudnorm.normalize.loudness(wav, loudness, TARGET_LUFS)
                peak = np.abs(wav).max()
                if peak > 0.99:  # avoid clipping after gain
                    wav = wav * (0.99 / peak)

                duration = len(wav) / TARGET_SAMPLE_RATE
                out_path = os.path.join(out_dir, os.path.basename(row["audio_path"]))
                sf.write(out_path, wav, TARGET_SAMPLE_RATE, subtype="PCM_16")
                yield {**row, "audio_path": out_path, "duration": duration}
            except Exception as e:
                print(f"Warning: standardize failed for {row['id']}: {e}. Skipping.")

    rows = read_manifest(args.work_dir, previous_stage("standardize"))
    results = run_spark_stage("standardize", rows, partition_fn, args.spark_master, args.num_partitions)
    write_manifest(args.work_dir, "standardize", results)
    print(
        "Human QA reminder: stratified-listen to ~100 clips across speakers/quality "
        f"bins from {out_dir} before training (see execution plan)."
    )


# ---------------------------------------------------------------------------
# Stage: text_process
# ---------------------------------------------------------------------------

HINDI_ONES = [
    "", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

HINDI_DIGITS = {"शून्य": 0}  # for readability of hindi_number_to_words below

HINDI_MONTHS = {
    1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल", 5: "मई", 6: "जून",
    7: "जुलाई", 8: "अगस्त", 9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर",
}

DEVANAGARI_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def hindi_number_to_words(n):
    """Convert a non-negative integer to Hindi words (Indian numbering system)."""
    if n == 0:
        return "शून्य"
    parts = []
    crore, n = divmod(n, 10000000)
    lakh, n = divmod(n, 100000)
    thousand, n = divmod(n, 1000)
    hundred, n = divmod(n, 100)
    if crore:
        parts.append(hindi_number_to_words(crore) + " करोड़")
    if lakh:
        parts.append(HINDI_ONES[lakh] + " लाख")
    if thousand:
        parts.append(HINDI_ONES[thousand] + " हज़ार")
    if hundred:
        parts.append(HINDI_ONES[hundred] + " सौ")
    if n:
        parts.append(HINDI_ONES[n])
    return " ".join(parts)


def expand_numerals(text):
    """Expand digits, dates, currency and common abbreviations into Hindi words."""
    text = text.translate(DEVANAGARI_DIGIT_MAP)

    def date_repl(m):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if month not in HINDI_MONTHS:
            return m.group(0)
        return f"{hindi_number_to_words(day)} {HINDI_MONTHS[month]} {hindi_number_to_words(year)}"

    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", date_repl, text)

    # ₹500 / ₹ 500 -> पाँच सौ रुपये
    text = re.sub(
        r"₹\s*(\d+)", lambda m: hindi_number_to_words(int(m.group(1))) + " रुपये", text
    )
    text = re.sub(r"(\d+)\s*%", lambda m: hindi_number_to_words(int(m.group(1))) + " प्रतिशत", text)

    # Decimals: 3.5 -> तीन दशमलव पाँच
    def decimal_repl(m):
        whole = hindi_number_to_words(int(m.group(1)))
        frac = " ".join(hindi_number_to_words(int(d)) for d in m.group(2))
        return f"{whole} दशमलव {frac}"

    text = re.sub(r"\b(\d+)\.(\d+)\b", decimal_repl, text)

    # Remaining plain integers
    text = re.sub(r"\d+", lambda m: hindi_number_to_words(int(m.group(0))), text)
    return text


def standardize_danda(text):
    """Use danda (।) as the sentence terminator; run after numeral expansion."""
    text = text.replace("|", "।")  # ASCII pipe often stands in for danda
    text = re.sub(r"\.(\s|$)", r"।\1", text)  # sentence-final period -> danda
    text = re.sub(r"।{2,}", "।", text)
    return text


LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_hindi_text(text, normalizer):
    """Full text pipeline for one utterance; returns None if it must be dropped."""
    text = normalizer.normalize(text)
    # NFC gives one canonical form per glyph; nukta letters (क़ ज़ फ़ ...) come out
    # consistently as base consonant + U+093C since precomposed forms are excluded
    # from Unicode composition.
    text = unicodedata.normalize("NFC", text)
    text = expand_numerals(text)
    text = standardize_danda(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or LATIN_RE.search(text):
        return None  # keep the corpus purely Devanagari + punctuation
    return text


def stage_text_process(args):
    def partition_fn(rows_iter):
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        normalizer = IndicNormalizerFactory().get_normalizer("hi")
        for row in rows_iter:
            text = normalize_hindi_text(row["text"], normalizer)
            if text is None:
                print(f"Filtered {row['id']}: empty or contains Latin script")
                continue
            yield {**row, "text": text}

    rows = read_manifest(args.work_dir, previous_stage("text_process"))
    results = run_spark_stage("text_process", rows, partition_fn, args.spark_master, args.num_partitions)
    write_manifest(args.work_dir, "text_process", results)


# ---------------------------------------------------------------------------
# Stage: build_vocab
# ---------------------------------------------------------------------------


def stage_build_vocab(args):
    rows = read_manifest(args.work_dir, previous_stage("build_vocab"))

    vocab_set = set()
    for row in rows:
        vocab_set.update(list(row["text"]))

    vocab_path = Path(args.work_dir) / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for ch in sorted(vocab_set):
            f.write(ch + "\n")
    print(f"Vocab size: {len(vocab_set)} -> {vocab_path}")
    if not 40 <= len(vocab_set) <= 120:
        print(
            f"Warning: vocab size {len(vocab_set)} is outside the expected ~70-90 range; "
            "inspect vocab.txt for stray symbols."
        )

    # Assert zero OOV characters before training (execution plan requirement)
    oov = {ch for row in rows for ch in row["text"] if ch not in vocab_set}
    assert not oov, f"OOV characters found: {sorted(oov)}"
    print("OOV check passed: every corpus character is in vocab.txt")

    write_manifest(args.work_dir, "build_vocab", rows)


# ---------------------------------------------------------------------------
# Stage: package (mel spectrograms -> Arrow shards + duration.json)
# ---------------------------------------------------------------------------


def stage_package(args):
    if not args.out_dir:
        raise ValueError("--out-dir is required for the package stage")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    def partition_fn(rows_iter):
        import soundfile as sf
        import torch

        from f5_tts.model.modules import MelSpec

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

    from datasets.arrow_writer import ArrowWriter
    from tqdm import tqdm

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STAGE_FUNCS = {
    "download": stage_download,
    "demix": stage_demix,
    "dereverb": stage_dereverb,
    "enhance": stage_enhance,
    "quality_filter": stage_quality_filter,
    "standardize": stage_standardize,
    "text_process": stage_text_process,
    "build_vocab": stage_build_vocab,
    "package": stage_package,
}


def get_args():
    parser = argparse.ArgumentParser(description="Prepare IndicVoices-R Hindi dataset for F5-TTS.")
    parser.add_argument("stage", choices=[*STAGE_ORDER, "all"], help="Pipeline stage to run.")
    parser.add_argument("--work-dir", required=True, help="Shared directory for intermediate audio + manifests.")
    parser.add_argument("--out-dir", help="Final dataset output directory (package stage).")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (download stage).")
    parser.add_argument("--spark-master", default=None, help='Spark master URL, e.g. "spark://host:7077" or "local[*]". Defaults to spark-submit config.')
    parser.add_argument("--num-partitions", type=int, default=DEFAULT_NUM_PARTITIONS, help="Spark partitions per stage.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N samples (smoke tests).")
    return parser.parse_args()


def cli():
    args = get_args()
    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n===== Running stage: {stage} =====")
        STAGE_FUNCS[stage](args)


if __name__ == "__main__":
    cli()
