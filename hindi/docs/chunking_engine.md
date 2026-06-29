# Milestone 3 – Smart Chunking Engine

## Objective

The goal of this milestone is to design and implement a **Smart Chunking Engine** that converts normalized Hindi text into **speech-friendly**, **model-friendly**, and **parallelizable** chunks.

This module serves as the bridge between the preprocessing pipeline and the TTS inference engine. It is one of the most critical components of the backend because every downstream module (Ray Scheduler, F5 Model, Audio Merger) depends on its output.

---

# Why Smart Chunking?

A naive implementation such as:

```python
text.split("।")
```

is insufficient because it ignores:

* Natural speech pauses
* Model input limitations
* Parallel inference requirements
* Audio quality

Instead, the chunker must intelligently divide text while preserving readability and speech naturalness.

---

# Responsibilities

The Smart Chunking Engine should:

* Accept normalized Hindi text.
* Generate natural speech boundaries.
* Ensure every chunk satisfies model input constraints.
* Preserve the original text order.
* Produce chunks that can be processed independently.
* Minimize the total number of model inference calls.

---

# Design Goals

## 1. Speech-Friendly

Chunks should end where humans naturally pause.

Good example:

```
भारत एक विशाल देश है।

यहाँ अनेक भाषाएँ बोली जाती हैं।
```

Bad example:

```
भारत एक वि

शाल देश है।
```

Words should never be split unless absolutely unavoidable.

---

## 2. Model-Friendly

Every TTS model has practical input limitations.

Instead of sending:

```
1000 character paragraph
```

the chunker should generate:

```
Chunk 1
Chunk 2
Chunk 3
...
```

Each chunk should remain below a configurable size limit.

Example:

```python
MAX_CHARS = 180
```

The value should be configurable rather than hardcoded.

This module is Hindi-specific, so split logic should use Hindi punctuation and Devanagari sentence boundaries instead of generic English-only rules.

---

## 3. Parallelizable

Generated chunks should be independent.

This enables Ray workers to synthesize multiple chunks simultaneously.

```
Chunk 1 → Worker 1

Chunk 2 → Worker 2

Chunk 3 → Worker 3
```

---

# Chunk Object

Instead of returning plain strings, the chunker should return structured objects.

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    id: int
    order: int
    text: str
    length: int
```

Future metadata can be added without changing the API.

Examples:

* audio_path
* duration
* status
* inference_time

---

# Splitting Strategy

The chunker should follow a hierarchical approach using Hindi-specific boundaries.

Priority:

```
Paragraph

↓

Sentence (Hindi punctuation: । , ? ! )

↓

Comma

↓

Semicolon

↓

Colon

↓

Whitespace

↓

Hard Split (Last Resort)
```

Preferred Hindi sentence boundaries include Devanagari punctuation such as `।`, `?`, and `!`.

Hard splitting should only occur when no natural Hindi boundary exists.

---

# Processing Pipeline

```
Normalized Text

↓

Paragraph Split

↓

Sentence Split

↓

Sentence Merge

↓

Oversized Sentence Split

↓

Chunk Objects
```

The engine should first split text into small logical units and then merge them into chunks close to the configured size limit.

---

# Sentence Merging

Very small sentences should be combined whenever possible.

Example

Input

```
Sentence 1 (35 chars)

Sentence 2 (30 chars)

Sentence 3 (20 chars)

Sentence 4 (40 chars)
```

Instead of generating four chunks,

the engine should produce:

```
Chunk 1

Sentence 1
Sentence 2
Sentence 3

Chunk 2

Sentence 4
```

This reduces the number of model invocations and improves throughput.

---

# Example

Suppose

```python
MAX_CHARS = 40
```

Input

```
भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं, और विभिन्न संस्कृतियाँ देखने को मिलती हैं।
```

Expected Output

```
Chunk 1

भारत एक विशाल देश है।

Chunk 2

यहाँ अनेक भाषाएँ बोली जाती हैं,

Chunk 3

और विभिन्न संस्कृतियाँ देखने को मिलती हैं।
```

---

# Public API

The module should expose a minimal interface.

```python
chunker = ChunkGenerator(max_chars=180)

chunks = chunker.chunk(text)
```

Output

```python
List[Chunk]
```

The public API should handle empty or whitespace-only text gracefully, either by returning an empty list or raising a clear Hindi-specific preprocessing exception.

No inference logic or audio generation should be part of this module.

---

# Design Principles

The Smart Chunking Engine should follow:

* Single Responsibility Principle
* Configurable chunk size
* Hindi-specific splitting rules
* Deterministic output
* Reusable API
* Independent unit testing

---

# Configuration

Example configuration

```python
ChunkGenerator(
    max_chars=180,
)
```

Future versions may support additional options such as:

* Minimum chunk length
* Preferred split characters
* Language-specific rules
* Model-specific chunk limits

---

# Separation of Responsibilities

The chunker should **only** generate chunks.

It must **not**:

* Generate audio
* Load models
* Perform inference
* Merge audio
* Schedule Ray tasks

Those responsibilities belong to later milestones.

---

# Architecture

```
Preprocessing

↓

Smart Chunking Engine

↓

Inference Scheduler (Ray)

↓

F5 Model Workers

↓

Audio Merger
```

---

# Testing Strategy

The implementation should satisfy the following invariants:

* No empty chunks
* No chunk exceeds `max_chars`
* Original order is preserved
* Combined chunk text reconstructs the normalized input
* Chunk count is minimized whenever possible
* Natural speech boundaries are preferred over hard splits

Unit tests should include:

* Short sentences
* Long paragraphs
* Mixed Hindi and English text
* Text without punctuation
* Extremely long sentences
* Multiple paragraphs
* Edge cases (empty text, whitespace-only input)

---

# Expected Deliverables

At the end of this milestone, the project should contain:

```
chunking/

    __init__.py

    chunk.py

    chunker.py

    constants.py

    exceptions.py

    README.md
```

The module should expose a reusable Smart Chunking Engine that produces ordered `Chunk` objects ready for parallel inference.

---

# Definition of Done

This milestone is considered complete when:

* Chunk objects are implemented.
* Configurable maximum chunk size is supported.
* Hierarchical splitting is implemented.
* Sentence merging is implemented.
* Oversized sentence handling is implemented.
* Public API is finalized.
* Unit tests pass.
* Documentation is complete.

---

# Next Milestone

The output of the Smart Chunking Engine will become the input to the **Ray-based Inference Scheduler**, where chunks will be distributed across multiple workers running the F5 Hindi TTS model.
