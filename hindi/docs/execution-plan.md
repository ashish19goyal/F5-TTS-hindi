# F5-TTS Hindi Execution Plan

## Dataset
- IndicVoices: A large-scale dataset of Indian languages, including Hindi, for text-to-speech (TTS) tasks. It contains high-quality recordings of native speakers and corresponding transcriptions.
    - 600+ hours of Hindi speech data
    - Low quality speech 
    - 16khz sampling rate
- IndicVoices-R: A refined version of the IndicVoices dataset, focusing on high-quality recordings and better coverage of various Hindi dialects.
    - 100+ hours of high-quality Hindi speech data
    - 22khz sampling rate
- IV-R: A subset of the IndicVoices-R dataset, specifically curated for TTS tasks, with a focus on diverse speaker representation and naturalness of speech.
    - 10+ hours of high-quality Hindi speech data
    - 22khz sampling rate

## Hardware
- GPU: 
    - NVIDIA RTX 5060 Ti (16Gb): 
        - 16GB GDDR6 memory
        - CUDA cores: 3584
        - Tensor cores: 112
        - Memory bandwidth: 448 GB/s
        - Use for early investigations, ablations, and small-scale experiments
    - NVIDIA A100 (80Gb): 
        - 80GB HBM2e memory
        - CUDA cores: 6912
        - Tensor cores: 432
        - Memory bandwidth: 1555 GB/s
        - Use for large-scale training and fine-tuning of the F5-TTS Hindi model

## Audio pre-processing
Raw indicVoices data is pre-processed to remove noise, silence, and irrelevant segments. IndicVoices-R data is already processed.
- Stage 1 — Demixing (background removal): Run HTDemucs (Hybrid Transformer Demucs) on each utterance to separate the vocal stem from background noise/music/chatter. Keep the vocal stem.
- Stage 2 — Dereverberation: Run VoiceFixer to reduce room reverb and restore degraded speech characteristics.
- Stage 3 — Speech enhancement: Run DeepFilterNet3 to suppress residual noise and artifacts.
- Stage 4 — Quality filtering (keep only what a TTS model should imitate)
    - SNR (e.g., WADA-SNR) — drop low-SNR residue.
    - C50 clarity (target comparable to IV-R's ~53 dB mean) — drops reverberant leftovers.
    - Perceptual MOS estimate (NORESQA-MOS or DNSMOS) — drop bottom tier.
    - Speaking rate and pitch variation within sane bounds — drops mumbled/degenerate clips.
    - Duration: keep 3–20 s.
- Stage 5 — Standardization
    - Resample to 24 kHz mono 16-bit WAV
    - Loudness-normalize to ≈ −23 to −20 LUFS; trim leading/trailing silence (>300 ms).
    - Human QA: stratified-random listen to ~100 clips across speakers/quality bins before committing to training.

## Text pre-processing
- Unicode NFC normalization (indic-nlp-library) — Devanagari has multiple encodings for identical glyphs; unnormalized text silently fragments your vocab.
- Expand numerals, dates, currency (₹), and abbreviations into Hindi words; 
- standardize danda (।) vs period; 
- make nukta usage (क़/ज़/फ़ …) consistent.
- drop utterances with Latin-script tokens; keeps the vocab purely Devanagari + punctuation.
- extract the character inventory of the final corpus → `vocab.txt`. Expect ~70–90 symbols (Devanagari block + matras + digits + punctuation). Assert zero OOV characters before training.

## Model Architecture
| Config | dim | depth | heads | Params | Role |
|---|---|---|---|---|---|
| Small | 768 | 18 | 12 | ~151M | **Primary** — proven from-scratch on Hindi at this data scale |
| Tiny | 512 | 12 | 8 | ~60–80M | Fallback / fast-iteration model for ablations; trains ~2–2.5× faster |
| Base | 1024 | 22 | 16 | ~336M | **Not recommended** from scratch on low volume of data. F5TTS base model trained on emilia dataset has these many params |

## Training configuration
| Aspect | Config | Comments |
|---|---|-----|
| Initialization | Random init |
| Learning rate | ~7.5e-5 with warmup (per F5 recipe) |
| Vocab | Build a clean Devanagari char vocab from own corpus |
| Data sensitivity | High — early training imprints data quality; noisy data teaches the model to generate noise |
| Sample rate / mel | 24 kHz, 100-dim, hop 256 | Must match Vocos vocoder |
| `batch_size_type` | `frame` | Efficient packing of variable-length clips |
| `batch_size_per_gpu` | 2,400 frames (start); 1,600 floor | Tune to VRAM |
| `grad_accumulation_steps` | 12–16 | Effective batch ≈ 29k–38k frames |
| Learning rate | 7.5e-5, warmup 10k–20k updates, then decay | F5 from-scratch recipe territory |
| `max_grad_norm` | 1.0 | |
| EMA | enabled | Evaluate/export EMA weights |
| Total updates | 300k target; 500k stretch |
| `save_per_updates` | 5,000 (+ frequent `last` autosave) | Crash resilience over multi-week runs |
| Audio length cap | 22 s | p99 of audio length in IndicVoices-R dataset |
