# Hindi Smart Chunking Engine

This package provides a Hindi-specific chunking engine for TTS preprocessing.

## Public API

```python
from pre_processing import HindiNormalizer
from chunking import ChunkGenerator

normalizer = HindiNormalizer()
clean_text = normalizer.normalize(text)

chunker = ChunkGenerator(max_chars=180)
chunks = chunker.chunk(clean_text)
```

## Chunk Rules

- Uses Hindi sentence boundaries such as `।`, `?`, and `!`
- Splits paragraphs on blank lines
- Merges short sentences into larger chunks without exceeding `max_chars`
- Falls back to whitespace-based hard splits only when needed
