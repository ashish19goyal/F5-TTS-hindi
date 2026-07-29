# Pipeline Integration Guide

End-to-end integration of three core modules:
1. **Pre-processing** (Text Normalization)
2. **Chunking** (Smart Text Segmentation)
3. **Inference** (Audio Generation)

## Architecture

```
Raw Hindi Text
    ↓
┌───────────────────────┐
│ Pre-Processing Module │  (HindiNormalizer)
│ - Unicode cleanup     │
│ - Whitespace fix      │
│ - Punctuation clean   │
└───────────────────────┘
    ↓ Normalized Text
┌───────────────────────┐
│ Chunking Module       │  (ChunkGenerator)
│ - Split by sentences  │
│ - Merge short chunks  │
│ - Balance sizes       │
└───────────────────────┘
    ↓ List of Chunks
┌───────────────────────┐
│ Inference Module      │  (MockInference / F5Inference)
│ - Generate audio      │
│ - Save WAV files      │
│ - Collect metadata    │
└───────────────────────┘
    ↓
Audio Results with Metadata
```

## Quick Start

### Basic Usage

```python
from pipeline import TTSPipeline

# Create pipeline
with TTSPipeline(max_chunk_chars=500) as pipeline:
    # Process text
    results = pipeline.process("नमस्ते, दुनिया!")
    
    # Results contain audio paths and metadata
    for result in results:
        print(f"Audio: {result.audio_path}")
        print(f"Duration: {result.duration}s")
```

### Run Demo Script

```bash
# Run comprehensive demo with multiple examples
python demo.py

# Run simple pipeline example
python pipeline.py

# Run integration tests
pytest tests/test_pipeline.py -v
```

## Components

### 1. Pre-Processing Module (`pre_processing/`)

**Purpose**: Normalize raw Hindi text

**Key Class**: `HindiNormalizer`

```python
from pre_processing.normlizer import HindiNormalizer

normalizer = HindiNormalizer()
clean_text = normalizer.normalize("कुछ  गन्दा   टेक्स्ट")
```

**Processing Steps**:
- Unicode normalization
- Invisible character removal
- Whitespace cleanup
- Quote normalization
- Punctuation cleanup

### 2. Chunking Module (`chunking/`)

**Purpose**: Split text into manageable chunks

**Key Class**: `ChunkGenerator`

```python
from chunking.chunker import ChunkGenerator

chunker = ChunkGenerator(max_chars=500)
chunks = chunker.chunk(normalized_text)

for chunk in chunks:
    print(f"Chunk {chunk.id}: {chunk.text}")
```

**Chunk Properties**:
- `id`: Unique identifier
- `order`: Sequential order
- `text`: Chunk content
- `length`: Character count

**Chunking Strategy**:
1. Split by paragraphs
2. Split by sentences (intelligent boundaries)
3. Merge short sentences
4. Split oversized chunks
5. Generate balanced chunks

### 3. Inference Module (`inference/`)

**Purpose**: Synthesize audio from text

**Key Class**: `MockInference` (with `BaseInference` interface)

```python
from inference import MockInference, Chunk as InferenceChunk

# Initialize
engine = MockInference(config={
    "output_dir": "./audio_output",
    "sample_rate": 22050,
})

# Load resources
engine.load_resources()

# Create chunk
chunk = InferenceChunk(
    chunk_id="chunk_001",
    text="नमस्ते दुनिया"
)

# Infer
result = engine.infer(chunk)
print(f"Audio saved to: {result.audio_path}")

# Cleanup
engine.cleanup()
```

## TTSPipeline Class

The main integration class that orchestrates all three modules.

### Initialization

```python
from pipeline import TTSPipeline

pipeline = TTSPipeline(
    max_chunk_chars=500,  # Maximum characters per chunk
    inference_config={     # Config for inference engine
        "output_dir": "./audio_output",
        "sample_rate": 22050,
        "duration": 1.0,
        "frequency": 440,
    }
)
```

### Methods

#### `process(raw_text: str) -> List[AudioResult]`

Process text through the entire pipeline.

```python
results = pipeline.process("नमस्ते दुनिया")
# Returns: List of AudioResult objects
```

**Returns**: `List[AudioResult]` containing:
- `chunk_id`: Identifier
- `audio_path`: Path to WAV file
- `duration`: Audio duration in seconds
- `sample_rate`: Sample rate in Hz
- `inference_latency`: Processing time
- `status`: 'success' or 'error'
- `metadata`: Dictionary with additional info

#### `cleanup()`

Release resources.

```python
pipeline.cleanup()
```

### Context Manager Usage

```python
with TTSPipeline() as pipeline:
    results = pipeline.process(text)
    # Automatically calls cleanup()
```

## Output Example

```
================================================================================
TTS PIPELINE EXECUTION SUMMARY
================================================================================

📝 INPUT TEXT (15 chars):
  नमस्ते, दुनिया!

✨ NORMALIZED TEXT (15 chars):
  नमस्ते, दुनिया!

📦 CHUNKS CREATED: 1

  Chunk 1:
    ID: chunk_001
    Audio Path: ./audio_output/chunk_001_647597be.wav
    Duration: 1.00s
    Sample Rate: 22050 Hz
    Latency: 1.2ms
    Status: success
    Metadata: {'order': 1, 'original_length': 15, 'model': 'MockInference', ...}

✅ PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

## Testing

### Test Suite

```bash
# Run all pipeline tests
pytest tests/test_pipeline.py -v

# Run specific test
pytest tests/test_pipeline.py::TestTTSPipeline::test_pipeline_single_chunk -v

# Run with coverage
pytest tests/test_pipeline.py --cov=pipeline
```

### Test Coverage

- Single chunk processing
- Multiple chunk processing
- Context manager usage
- Metadata completeness
- Special characters handling
- Long text processing
- Resource lifecycle

## Configuration Options

### TTSPipeline Config

```python
config = {
    "max_chunk_chars": 500,      # Chunking limit
    "output_dir": "./audio",     # Audio output directory
    "sample_rate": 22050,        # Hz
    "duration": 1.0,             # Seconds
    "frequency": 440,            # Hz (sine wave)
}
```

## Module Interfaces

### ChunkGenerator to InferenceChunk Conversion

The pipeline converts chunks from the chunking module to the inference module:

```python
# From chunking module
chunking_chunk = Chunk(
    id=1,
    order=1,
    text="नमस्ते",
    length=6
)

# To inference module
inference_chunk = InferenceChunk(
    chunk_id="chunk_001",
    text=chunking_chunk.text,
    metadata={
        "order": chunking_chunk.order,
        "original_length": chunking_chunk.length,
    }
)
```

## Error Handling

```python
from inference import (
    InferenceException,
    InvalidChunkError,
    AudioGenerationError,
    ResourceLoadError,
)

try:
    pipeline = TTSPipeline()
    results = pipeline.process(text)
except InvalidChunkError as e:
    print(f"Invalid input: {e}")
except AudioGenerationError as e:
    print(f"Audio generation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    pipeline.cleanup()
```

## Performance Notes

### MockInference

- **Latency**: 1-2ms per chunk
- **Memory**: ~50MB
- **GPU**: Not required

### System Requirements

- Python 3.8+
- numpy, scipy (for audio processing)
- pytest, pytest-asyncio (for testing)

## Future Integration

### Switching to Real F5 Model

```python
# Instead of MockInference
from inference import MockInference

# Will become
from inference import F5Inference

# Same interface - no other code changes needed!
pipeline = TTSPipeline(
    inference_config={"model": "F5"}
)
```

## Troubleshooting

### Audio Files Not Generated

1. Check `output_dir` has write permissions
2. Verify `load_resources()` was called
3. Check chunk validation (non-empty text)

### Import Errors

```bash
# Ensure you're in the backend directory
cd src/backend

# Verify modules are in Python path
python -c "import pre_processing, chunking, inference; print('OK')"
```

### Metadata Missing

- Order and original_length metadata are added by pipeline
- Model-specific metadata from inference is preserved
- Both are merged in result.metadata

## Files

### Main Scripts
- `pipeline.py` - TTSPipeline class and utilities
- `demo.py` - Comprehensive demo with examples
- `tests/test_pipeline.py` - Integration tests

### Module Dependencies
- `pre_processing/normlizer.py` - Text normalization
- `chunking/chunker.py` - Text chunking
- `inference/` - Audio generation

## Next Steps

1. **Milestone 5**: Ray-based Distributed Scheduler
   - Distribute chunks across Ray workers
   - Parallel inference execution

2. **Milestone 6**: F5 Model Integration
   - Replace MockInference with F5Inference
   - Real Hindi TTS synthesis

3. **Milestone 7**: API Layer
   - FastAPI REST endpoints
   - WebSocket streaming
   - File upload/download

## Quick Reference

```python
# Basic pipeline usage
from pipeline import TTSPipeline

with TTSPipeline() as pipeline:
    text = "नमस्ते दुनिया"
    results = pipeline.process(text)
    
    for result in results:
        print(f"✅ {result.chunk_id}: {result.audio_path}")

# With custom config
pipeline = TTSPipeline(
    max_chunk_chars=300,
    inference_config={
        "output_dir": "./my_audio",
        "sample_rate": 44100,
    }
)

# Run tests
# pytest tests/test_pipeline.py -v

# Run demo
# python demo.py
```

---

**Last Updated**: 2026-06-29  
**Status**: ✅ Complete and Tested
