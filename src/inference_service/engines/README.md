# Inference Abstraction Layer

Model-agnostic inference interface for TTS synthesis.

## Overview

The inference layer acts as the abstraction between the Smart Chunking Engine and any TTS model implementation. This design allows swapping models without modifying the core pipeline.

## Architecture

```
Chunk
  ↓
Inference Interface
  ├─ MockInference
  ├─ F5Inference (future)
  ├─ XTTSInference (future)
  └─ BarkInference (future)
  ↓
AudioResult
```

## Usage

### Basic Example

```python
from inference import MockInference, Chunk

# Create inference engine
engine = MockInference(config={
    "output_dir": "./audio_output",
    "sample_rate": 22050,
    "duration": 1.0
})

# Load resources
engine.load_resources()

# Create a chunk
chunk = Chunk(
    chunk_id="chunk_001",
    text="नमस्ते दुनिया",
    metadata={"speaker_id": "speaker_001"}
)

# Run inference
result = engine.infer(chunk)

print(f"Audio saved to: {result.audio_path}")
print(f"Duration: {result.duration} seconds")
print(f"Latency: {result.inference_latency:.3f} seconds")

# Cleanup
engine.cleanup()
```

### Asynchronous Usage with Ray

```python
import ray
from inference import MockInference

@ray.remote
class InferenceWorker:
    def __init__(self):
        self.engine = MockInference()
        self.engine.load_resources()
    
    async def process_chunk(self, chunk):
        return await self.engine.infer_async(chunk)

# Submit work to Ray
worker = InferenceWorker.remote()
result = ray.get(worker.process_chunk.remote(chunk))
```

## Components

### Models

- **Chunk**: Input text and metadata
- **AudioResult**: Output audio path and metadata

### Exceptions

- `InferenceException`: Base exception
- `InvalidChunkError`: Invalid chunk input
- `AudioGenerationError`: Audio generation failure
- `ResourceLoadError`: Resource initialization failure

### Implementations

#### MockInference

Test implementation that generates sine wave audio.

**Config Options:**
- `sample_rate`: Output sample rate (default: 22050)
- `duration`: Audio duration in seconds (default: 1.0)
- `frequency`: Sine wave frequency (default: 440 Hz)
- `output_dir`: Audio output directory (default: "./audio_output")

**Features:**
- Deterministic output (same text → same audio)
- No GPU required
- Fast synthesis
- Valid WAV file output

## Interface Specification

All implementations must inherit from `BaseInference` and implement:

```python
class BaseInference(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None): ...
    
    @abstractmethod
    def infer(self, chunk: Chunk) -> AudioResult: ...
    
    @abstractmethod
    async def infer_async(self, chunk: Chunk) -> AudioResult: ...
    
    @abstractmethod
    def load_resources(self) -> None: ...
    
    @abstractmethod
    def cleanup(self) -> None: ...
```

## Interface Validation

Validate implementations at runtime:

```python
from inference import InferenceInterfaceValidator, MockInference

engine = MockInference()
InferenceInterfaceValidator.validate(engine)  # Raises if invalid
```

## Testing

Run the test suite:

```bash
pytest tests/test_inference.py -v
```

Tests verify:
- Chunk validation
- Audio file generation
- Metadata completeness
- Exception handling
- Resource lifecycle
- Synchronous and asynchronous operations
- Interface compliance

## Future Implementations

Planned inference engines:

1. **F5Inference**: Official F5 TTS Hindi model
2. **XTTSInference**: Coqui XTTS multilingual
3. **BarkInference**: Suno Bark generative model

Adding new implementations:

1. Create `f5_inference.py` inheriting from `BaseInference`
2. Implement all required methods
3. Add tests in `tests/`
4. Update `__init__.py` exports
5. Update this README

No changes needed to existing code!

## Best Practices

1. **Always call `load_resources()`** before inference
2. **Always call `cleanup()`** on shutdown for proper resource cleanup
3. **Validate chunks** before inference
4. **Use `infer_async()`** with Ray workers for distributed execution
5. **Handle specific exceptions** (`InvalidChunkError`, `AudioGenerationError`, etc.)

## Performance

MockInference benchmarks:
- Synthesis time: ~10-50ms per chunk
- Memory: ~50MB
- GPU: Not required

(F5Inference metrics to be updated after integration)
