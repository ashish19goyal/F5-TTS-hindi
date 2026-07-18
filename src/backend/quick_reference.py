#!/usr/bin/env python
"""
Quick Reference: TTS Pipeline

Usage Examples:
    python quick_reference.py
"""

# ============================================================================
# EXAMPLE 1: Basic Usage
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Basic Pipeline Usage")
print("="*80)

code_example_1 = """
from backend.pipeline import TTSPipeline

with TTSPipeline() as pipeline:
    text = "नमस्ते, दुनिया!"
    results = pipeline.process(text)
    
    for result in results:
        print(f"✅ {result.chunk_id}: {result.audio_path}")
"""

print(code_example_1)

# ============================================================================
# EXAMPLE 2: With Configuration
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: Custom Configuration")
print("="*80)

code_example_2 = """
from backend.pipeline import TTSPipeline

config = {
    "output_dir": "./my_audio",
    "sample_rate": 44100,
    "duration": 1.5,
}

pipeline = TTSPipeline(
    max_chunk_chars=300,
    inference_config=config
)

pipeline.load_resources()
results = pipeline.process(long_text)
pipeline.cleanup()
"""

print(code_example_2)

# ============================================================================
# EXAMPLE 3: Processing Multiple Texts
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 3: Process Multiple Texts")
print("="*80)

code_example_3 = """
from backend.pipeline import TTSPipeline

texts = [
    "नमस्ते दुनिया",
    "मेरा नाम राज है",
    "प्रोग्रामिंग बहुत मजेदार है",
]

with TTSPipeline() as pipeline:
    for text in texts:
        results = pipeline.process(text)
        print(f"✅ Generated {len(results)} audio files")
"""

print(code_example_3)

# ============================================================================
# EXAMPLE 4: Working with Results
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 4: Process Results")
print("="*80)

code_example_4 = """
from backend.pipeline import TTSPipeline
from pathlib import Path

with TTSPipeline() as pipeline:
    results = pipeline.process(text)
    
    total_duration = 0
    total_latency = 0
    
    for result in results:
        # Check result status
        if result.status == "success":
            # Audio file information
            audio_file = Path(result.audio_path)
            file_size = audio_file.stat().st_size
            
            # Timing information
            total_duration += result.duration
            total_latency += result.inference_latency
            
            # Metadata access
            order = result.metadata.get("order", "N/A")
            model = result.metadata.get("model", "N/A")
            
            print(f"Chunk {order}: {file_size} bytes, {result.duration}s")
    
    print(f"Total: {total_duration}s audio, {total_latency*1000:.1f}ms latency")
"""

print(code_example_4)

# ============================================================================
# EXAMPLE 5: Error Handling
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 5: Error Handling")
print("="*80)

code_example_5 = """
from backend.pipeline import TTSPipeline
from backend.inference import (
    InferenceException,
    InvalidChunkError,
    AudioGenerationError,
)

try:
    with TTSPipeline() as pipeline:
        results = pipeline.process(text)
        
except InvalidChunkError as e:
    print(f"❌ Invalid chunk: {e}")
    
except AudioGenerationError as e:
    print(f"❌ Audio generation failed: {e}")
    
except InferenceException as e:
    print(f"❌ Inference error: {e}")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
"""

print(code_example_5)

# ============================================================================
# EXAMPLE 6: Using Individual Modules
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 6: Using Individual Modules")
print("="*80)

code_example_6 = """
# Step 1: Normalization
from backend.pre_processing.normlizer import HindiNormalizer

normalizer = HindiNormalizer()
clean_text = normalizer.normalize(raw_text)

# Step 2: Chunking
from backend.chunking.chunker import ChunkGenerator

chunker = ChunkGenerator(max_chars=500)
chunks = chunker.chunk(clean_text)

# Step 3: Inference
from backend.inference import MockInference, Chunk as InferenceChunk

engine = MockInference()
engine.load_resources()

results = []
for chunk in chunks:
    inference_chunk = InferenceChunk(
        chunk_id=f"chunk_{chunk.id:03d}",
        text=chunk.text
    )
    result = engine.infer(inference_chunk)
    results.append(result)

engine.cleanup()
"""

print(code_example_6)

# ============================================================================
# QUICK REFERENCE TABLE
# ============================================================================
print("\n" + "="*80)
print("QUICK REFERENCE TABLE")
print("="*80)

reference = """
╔════════════════════╦════════════════════════════════════════════════════╗
║ Task               ║ Code                                               ║
╠════════════════════╬════════════════════════════════════════════════════╣
║ Basic usage        ║ with TTSPipeline() as p: p.process(text)          ║
║ Custom output dir  ║ TTSPipeline(inference_config={"output_dir": ...}) ║
║ Smaller chunks     ║ TTSPipeline(max_chunk_chars=300)                   ║
║ Access audio path  ║ result.audio_path                                  ║
║ Get duration       ║ result.duration                                    ║
║ Get latency        ║ result.inference_latency                           ║
║ Get status         ║ result.status                                      ║
║ Get metadata       ║ result.metadata                                    ║
║ Normalize text     ║ HindiNormalizer().normalize(text)                  ║
║ Create chunks      ║ ChunkGenerator().chunk(text)                       ║
║ Run inference      ║ MockInference().infer(chunk)                       ║
║ Async inference    ║ await engine.infer_async(chunk)                    ║
╚════════════════════╩════════════════════════════════════════════════════╝
"""

print(reference)

# ============================================================================
# AVAILABLE COMMANDS
# ============================================================================
print("\n" + "="*80)
print("AVAILABLE COMMANDS")
print("="*80)

commands = """
📊 Run Tests:
   pytest tests/test_pipeline.py -v      # Pipeline tests
   pytest tests/test_inference.py -v     # Inference tests
   pytest tests/ -v                      # All tests

🎬 Run Demos:
   python pipeline.py                    # Simple demo
   python demo.py                        # Comprehensive demo
   python quick_reference.py             # This file

📖 Read Documentation:
   INTEGRATION_SUMMARY.md                # Overview
   PIPELINE_INTEGRATION.md               # Detailed guide
   inference/README.md                   # Inference module
"""

print(commands)

# ============================================================================
# MODULE RESPONSIBILITIES
# ============================================================================
print("\n" + "="*80)
print("MODULE RESPONSIBILITIES")
print("="*80)

responsibilities = """
Pre-Processing Module (normlizer.py)
  ✓ Unicode normalization
  ✓ Remove invisible characters
  ✓ Clean whitespace
  ✓ Normalize quotes
  ✓ Clean punctuation
  Input: Raw Hindi text
  Output: Normalized Hindi text

Chunking Module (chunker.py)
  ✓ Split by paragraphs
  ✓ Split by sentences
  ✓ Merge short chunks
  ✓ Balance chunk sizes
  Input: Normalized text
  Output: List[Chunk]

Inference Module (inference/)
  ✓ Generate audio from text
  ✓ Save WAV files
  ✓ Collect metadata
  ✓ Handle async processing
  Input: Text chunk
  Output: AudioResult with file path
"""

print(responsibilities)

# ============================================================================
# CONFIGURATION OPTIONS
# ============================================================================
print("\n" + "="*80)
print("CONFIGURATION OPTIONS")
print("="*80)

config_guide = """
TTSPipeline Options:
  max_chunk_chars: int                  # Default: 500
    Maximum characters per chunk

  inference_config: Dict[str, Any]      # Default: {...}
    output_dir: str                     # Audio output directory
    sample_rate: int                    # Default: 22050 Hz
    duration: float                     # Default: 1.0 seconds
    frequency: int                      # Default: 440 Hz (sine wave)

Examples:
  TTSPipeline()                                    # All defaults
  TTSPipeline(max_chunk_chars=300)                # Smaller chunks
  TTSPipeline(inference_config={"output_dir": "."}) # Custom directory
"""

print(config_guide)

# ============================================================================
# TROUBLESHOOTING
# ============================================================================
print("\n" + "="*80)
print("TROUBLESHOOTING")
print("="*80)

troubleshooting = """
❓ ImportError: No module named 'pre_processing'
   ✅ Make sure you're in src/backend/ directory

❓ Audio files not generated
   ✅ Check output_dir has write permissions
   ✅ Verify load_resources() was called
   ✅ Check text is not empty

❓ Tests failing
   ✅ pytest tests/test_pipeline.py -v
   ✅ pip install pytest pytest-asyncio scipy numpy

❓ Module not found errors
   ✅ cd src/backend
   ✅ python pipeline.py

❓ Out of memory
   ✅ Reduce max_chunk_chars
   ✅ Process texts in batches

❓ Slow processing
   ✅ Increase max_chunk_chars (fewer chunks)
   ✅ Use MockInference instead of real model
"""

print(troubleshooting)

# ============================================================================
# COMMON PATTERNS
# ============================================================================
print("\n" + "="*80)
print("COMMON PATTERNS")
print("="*80)

patterns = """
Pattern 1: Process and Save Results
  with TTSPipeline() as pipeline:
      results = pipeline.process(text)
      for r in results:
          print(f"✅ Saved: {r.audio_path}")

Pattern 2: Batch Processing
  with TTSPipeline() as pipeline:
      for text in text_list:
          results = pipeline.process(text)

Pattern 3: Monitor Progress
  with TTSPipeline() as pipeline:
      results = pipeline.process(text)
      for i, r in enumerate(results, 1):
          print(f"[{i}/{len(results)}] {r.status}")

Pattern 4: Error Recovery
  with TTSPipeline() as pipeline:
      try:
          results = pipeline.process(text)
      except Exception as e:
          print(f"Error: {e}, continuing...")
          results = []

Pattern 5: Performance Monitoring
  with TTSPipeline() as pipeline:
      results = pipeline.process(text)
      latencies = [r.inference_latency for r in results]
      print(f"Avg latency: {sum(latencies)/len(latencies)*1000:.1f}ms")
"""

print(patterns)

print("\n" + "="*80)
print("✅ For more examples, see demo.py or PIPELINE_INTEGRATION.md")
print("="*80 + "\n")
