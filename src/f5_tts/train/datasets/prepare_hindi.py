"""
Usage:
    python prepare_hindi.py huggingFaceLoginToken /output/dataset/path [--pretrain] [--workers N]
"""

import concurrent.futures
import multiprocessing
import os
import signal
import sys
from contextlib import contextmanager
from huggingface_hub import login

sys.path.append(os.getcwd())

import argparse
import json
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
from datasets import load_dataset

# Configuration constants
BATCH_SIZE = 100  # Batch size for text conversion
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 100  # Number of files to process per worker batch
executor = None  # Global executor for cleanup


@contextmanager
def graceful_exit():
    """Context manager for graceful shutdown on signals"""

    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if executor is not None:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def process_audio(audio_file, text, out_dir, chunk, sample):
    """Process a single audio and extracting duration."""
    try:
        audio_array = audio_file["array"]
        sampling_rate = audio_file["sampling_rate"]
        audio_duration = len(audio_array)/sampling_rate
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
        
        wav_path = os.path.abspath(f"{out_dir}/wavs/hindi_{chunk:05d}_{sample:05d}.wav")
        sf.write(wav_path, audio_array, sampling_rate)
        return (wav_path, text, audio_duration)
    except Exception as e:
        print(f"Warning: Failed to process {wav_path} due to error: {e}. Skipping corrupt sample.")
        return None


def download_indic_voices_dataset():
    ## Import indicVoices dataset for hindi language
    dataset = load_dataset("ai4bharat/IndicVoices", "hindi", split="train")
    # Keep only audio_filepath and text columns
    dataset = dataset.select_columns(["audio_filepath", "text"])
    print("Downloaded Indic Voices dataset")
    print(dataset)
    return dataset

def prepare_training_data(huggingface_token, out_dir, num_workers=None):
    global executor

    # Hugging face login to download dataset
    login(huggingface_token)

    indic_voices_dataset = download_indic_voices_dataset()
    total_data_size = len(indic_voices_dataset)
    
    # Use provided worker count or calculate optimal number
    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_data_size)
    print(f"\nProcessing {total_data_size} audio using {worker_count} workers...")

    with graceful_exit():
        # Initialize thread pool with optimized settings
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            results = []

            # Process files in chunks for better efficiency
            for i in range(0, total_data_size-CHUNK_SIZE, CHUNK_SIZE):
                chunk_futures = []
                for j in range(CHUNK_SIZE):
                    # Submit futures in order
                    chunk_futures.append(executor.submit(process_audio, indic_voices_dataset[i+j]['audio_filepath'], indic_voices_dataset[i+j]['text'], out_dir, i, j))

                # Iterate over futures in the original submission order to preserve ordering
                for future in tqdm(
                    chunk_futures,
                    total=CHUNK_SIZE,
                    desc=f"Processing chunk {i // CHUNK_SIZE + 1}/{(total_data_size + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing file: {e}")

            executor = None

    # Filter out failed results
    processed = [res for res in results if res is not None]
    if not processed:
        raise RuntimeError("No valid audio files were processed!")

    # Batch process text conversion
    raw_texts = [item[1] for item in processed]

    # Prepare final results
    sub_result = []
    durations = []
    vocab_set = set()

    for (audio_path, _, duration), raw_text in zip(processed, raw_texts):
        sub_result.append({"audio_path": audio_path, "text": raw_text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(raw_text))

    return sub_result, durations, vocab_set


def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix()) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # Save durations to JSON
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Handle vocab file - write only once based on finetune flag
    voca_out_path = out_dir / "vocab.txt"
    with open(voca_out_path.as_posix(), "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


def prepare_and_save_set(huggingface_token, out_dir, num_workers: int = None):
    sub_result, durations, vocab_set = prepare_training_data(huggingface_token, out_dir, num_workers=num_workers)
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set)


def get_args():
    parser = argparse.ArgumentParser(description="Prepare and save dataset.")
    parser.add_argument(
        "huggingface_token",
        type=str,
        help="Login token for hugging face.",
    )
    parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
    parser.add_argument("--pretrain", action="store_true", help="Enable for new pretrain, otherwise is a fine-tune")
    parser.add_argument("--workers", type=int, help=f"Number of worker threads (default: {MAX_WORKERS})")
    return parser.parse_args()


def cli():
    try:
        args = get_args()
        prepare_and_save_set(args.huggingface_token, args.out_dir, num_workers=args.workers)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
