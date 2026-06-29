# Milestone 4 – Inference Abstraction Layer

## Objective

The objective of this milestone is to design and implement a **Model-Agnostic Inference Layer** that acts as the interface between the Smart Chunking Engine and the underlying Text-to-Speech model.

Instead of tightly coupling the backend to the F5 Hindi TTS model, the system introduces an abstraction layer that allows any compatible TTS model to be plugged into the pipeline with minimal code changes.

For this milestone, a **Mock Inference Engine** will be implemented. The actual F5 model integration will be completed in a later milestone.

---

# Motivation

The backend should not depend directly on a specific AI model.

Instead of:

```text
Chunk
   │
   ▼
F5 Model
```

the architecture becomes:

```text
Chunk
   │
   ▼
Inference Interface
   │
   ├──────────────┐
   ▼              ▼
Mock Model     F5 Model
```

This approach provides:

* Loose coupling
* Better maintainability
* Easier testing
* Model interchangeability
* Cleaner software architecture

---

# Architecture

```text
Normalized Text
        │
        ▼
Smart Chunking Engine
        │
        ▼
Inference Interface
        │
        ├──────────────┐
        ▼              ▼
Mock Inference     F5 Inference
        │
        ▼
Audio Result
```

The scheduler, API, and audio processing modules interact only with the **Inference Interface**, never with a specific model implementation.

---

# Responsibilities

The inference layer is responsible for:

* Accepting a single chunk.
* Running TTS inference.
* Returning synthesized audio.
* Returning inference metadata.

The inference layer is **not** responsible for:

* Text normalization
* Chunk generation
* Audio merging
* Ray scheduling
* API routing

Each component should have a single responsibility.

---

# Design Principles

The inference layer follows:

* Single Responsibility Principle
* Open/Closed Principle
* Dependency Inversion Principle

The backend depends on an abstraction rather than a concrete implementation.

---

# Public Interface

Every inference implementation should expose the same public API.

## Core Methods

```python
class BaseInference:
    # Initialization with configuration
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the inference engine.
        
        Args:
            config: Optional configuration dictionary with defaults.
                   Keys may include: model_name, device, batch_size, etc.
        """
        pass
    
    # Synchronous inference
    def infer(self, chunk: Chunk) -> AudioResult:
        """
        Synchronous inference for a single chunk.
        
        Args:
            chunk: The text chunk to synthesize
            
        Returns:
            AudioResult object with audio file path and metadata
        """
        pass
    
    # Asynchronous inference for Ray workers
    async def infer_async(self, chunk: Chunk) -> AudioResult:
        """
        Asynchronous inference for distributed execution.
        
        Args:
            chunk: The text chunk to synthesize
            
        Returns:
            AudioResult object with audio file path and metadata
        """
        pass
```

## Resource Lifecycle

```python
    def load_resources(self) -> None:
        """
        Load and initialize resources (e.g., model to GPU).
        Called once during startup.
        """
        pass
    
    def cleanup(self) -> None:
        """
        Release resources and clean up (e.g., free GPU memory).
        Called during shutdown.
        """
        pass
```

Regardless of the implementation, the caller should never need to know which model is being used.

---

# Mock Inference Engine

The first implementation will be a **Mock Inference Engine**.

Its purpose is to validate the backend pipeline before integrating the real F5 model.

Instead of synthesizing speech, it will generate predictable placeholder audio.

Possible outputs include:

* One second of silence
* A sine wave tone
* A generated WAV file
* Dummy audio bytes

The important point is that every invocation returns a valid audio object.

---

# Why Use a Mock Model?

Using a mock implementation provides several advantages:

* Backend can be developed independently of the AI model.
* Unit testing becomes significantly easier.
* API development does not depend on GPU availability.
* Ray integration can be completed before model integration.
* Failures become easier to isolate.

This significantly reduces development complexity.

---

# Future Model Integration

When the F5 model becomes available, only a new implementation needs to be created.

Current architecture:

```text
Chunk

↓

MockInference
```

Future architecture:

```text
Chunk

↓

F5Inference
```

No other modules require modification.

The scheduler, API, and audio merger continue using the same interface.

---

# Error Handling

The inference layer defines specific exceptions for error handling:

```python
class InferenceException(Exception):
    """Base exception for all inference-related errors."""
    pass

class InvalidChunkError(InferenceException):
    """Raised when chunk validation fails."""
    pass

class AudioGenerationError(InferenceException):
    """Raised when audio generation fails."""
    pass

class ResourceLoadError(InferenceException):
    """Raised when resource loading fails."""
    pass
```

Implementations should catch and re-raise as appropriate exceptions, providing clear error messages.

---

# Suggested Project Structure

```text
inference/

    __init__.py

    base.py

    mock_inference.py

    models.py

    exceptions.py

    interface_validator.py

    README.md
```

Future additions:

```text
f5_inference.py

xtts_inference.py

bark_inference.py
```

No existing code should require modification when new models are added.

---

# Data Flow

```text
Chunk

↓

Inference Interface

↓

Audio Result
```

Each chunk is processed independently.

This design naturally supports distributed execution in future milestones.

---

# Audio Result

Instead of returning raw bytes directly, inference should return a structured result object.

## AudioResult Structure

```python
class AudioResult:
    chunk_id: str              # Unique identifier for the chunk
    audio_path: str            # Path to WAV file containing synthesized audio
    duration: float            # Audio duration in seconds
    sample_rate: int           # Sample rate in Hz (typically 22050 or 44100)
    inference_latency: float   # Time taken for inference in seconds
    status: str                # Status: 'success', 'error', etc.
    metadata: Dict[str, Any]   # Additional metadata (language, speaker_id, etc.)
```

## Design Notes

* Audio is stored as **WAV files** on disk rather than in-memory bytes to minimize memory usage
* Each inference result gets a unique file path
* AudioResult provides structured metadata for logging and monitoring
* Paths support both local and cloud storage (S3, etc.) in future iterations

Using a structured object makes future extensions much easier.

---

# Interface Validator

A minimal interface validator ensures all implementations conform to the contract:

```python
class InferenceInterfaceValidator:
    """
    Validates that an inference implementation meets interface requirements.
    """
    @staticmethod
    def validate(inference_engine: BaseInference) -> bool:
        """
        Check that engine has required methods and lifecycle support.
        Returns True if valid, raises InferenceException otherwise.
        """
        required_methods = ['infer', 'infer_async', 'load_resources', 'cleanup']
        for method in required_methods:
            if not hasattr(inference_engine, method):
                raise InferenceException(f"Missing required method: {method}")
        return True
```

This minimal validation ensures compatibility without being too prescriptive.

---

# Testing Strategy

The mock implementation should satisfy the following:

* Accept valid chunk objects.
* Return valid audio output (WAV files).
* Produce deterministic results.
* Return complete metadata.
* Handle invalid inputs gracefully.
* Support both sync and async operations.
* Implement resource lifecycle methods.

Unit tests should verify:

* Successful inference (sync and async).
* Invalid chunk handling.
* Audio file generation and paths.
* Metadata completeness.
* Exception handling.
* Resource lifecycle (load_resources and cleanup).
* Interface validator compliance.

---

# Benefits

The abstraction layer provides:

* Decoupled architecture
* Easier maintenance
* Better testing
* Improved scalability
* Simpler future model integration
* Cleaner project organization

Most importantly, it enables the backend to evolve independently from the AI model.

---

# Definition of Done

This milestone is complete when:

* A model-independent inference interface exists with all core methods.
* Configuration support via `__init__` with defaults is implemented.
* Resource lifecycle methods (load_resources/cleanup) are implemented.
* Both sync and async inference methods are implemented.
* Specific exceptions (InferenceException, InvalidChunkError, AudioGenerationError, ResourceLoadError) are defined.
* AudioResult returns WAV file paths with complete metadata.
* Interface validator is implemented and working.
* Mock inference implementation is complete.
* Unit tests pass (sync, async, lifecycle, validation, exceptions).
* Documentation is complete.

---

# Next Milestone

The next milestone introduces the **Ray-based Distributed Scheduler**.

Instead of invoking the inference engine directly, chunks will be dispatched to multiple Ray workers running the same inference interface.

The scheduler will remain completely independent of the underlying model implementation, demonstrating the effectiveness of the abstraction introduced in this milestone.
