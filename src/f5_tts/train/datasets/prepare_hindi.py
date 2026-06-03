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
from f5_tts.model.modules import MelSpec
import torch

# Configuration constants
BATCH_SIZE = 100  # Batch size for text conversion
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 10000  # Number of files to process per worker batch
executor = None  # Global executor for cleanup
mel_spec = MelSpec(target_sample_rate=16000)

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


def process_audio(audio_file, text):
    """Process a single audio and extracting duration."""
    try:
        audio_array = audio_file["array"]
        sampling_rate = audio_file["sampling_rate"]
        audio_duration = len(audio_array)/sampling_rate
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
        
        # convert to tensor and change shape to (1,len) for mel processing
        audio_array = torch.tensor(audio_array).reshape([1,-1])
        mel_spectrograms = mel_spec(audio_array)    
        mel_spectrograms = mel_spectrograms.squeeze(0)

        return (mel_spectrograms, text, audio_duration)
    except Exception as e:
        print(f"Warning: Failed to process {text} due to error: {e}. Skipping corrupt sample.")
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

    durations = []
    vocab_set = set()

    with graceful_exit():
        # Initialize thread pool with optimized settings
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            
            # Process files in chunks for better efficiency
            for i in range(0, total_data_size-CHUNK_SIZE, CHUNK_SIZE):
                chunk_futures = []
                chunk_results = []
                for j in range(CHUNK_SIZE):
                    # Submit futures in order
                    chunk_futures.append(executor.submit(process_audio, indic_voices_dataset[i+j]['audio_filepath'], indic_voices_dataset[i+j]['text']))

                # Iterate over futures in the original submission order to preserve ordering
                for future in tqdm(
                    chunk_futures,
                    total=CHUNK_SIZE,
                    desc=f"Processing chunk {i // CHUNK_SIZE + 1}/{(total_data_size + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            chunk_results.append(result)
                    except Exception as e:
                        print(f"Error processing file: {e}")

                # Filter out failed results
                processed_chunk_results = [res for res in chunk_results if res is not None]
                if not processed_chunk_results:
                    raise RuntimeError("No valid audio files were processed in chunk {i}!")
                
                arrow_object = []
                for mel_spectrograms, raw_text, duration in processed_chunk_results:
                    arrow_object.append({"mel_spec": mel_spectrograms, "text": raw_text, "duration": duration})
                    durations.append(duration)
                    vocab_set.update(list(raw_text))
                save_mels(out_dir, arrow_object, i)

            executor = None    
    
    save_duration_vocab(out_dir, durations, vocab_set)


def save_duration_vocab(out_dir, duration_list, text_vocab_set):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

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
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")

def save_mels(out_dir, result, file_index):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    raw_arrow_path = out_dir / f"mel_{file_index}.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix()) as writer:
        for line in tqdm(result, desc="Writing to mel.arrow ..."):
            writer.write(line)
        writer.finalize()

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")


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
        prepare_training_data(args.huggingface_token, args.out_dir, num_workers=args.workers)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
