# Hindi Text Normalization

## Overview

This module performs preprocessing on Hindi text before it is passed to the Text-to-Speech inference pipeline.

The goal is to produce clean and standardized input without altering the semantic meaning of the original text.

---

## Pipeline - Will Break this into small modules in Next phase

```
Raw Text
    │
    ▼
Unicode Normalization
    ▼
Invisible Character Removal
    ▼
Whitespace Normalization
    ▼
Quote Normalization
    ▼
Punctuation Normalization
    ▼
Trim
    ▼
Normalized Text
```

---

## Features

### Unicode Normalization

Uses Unicode NFC normalization to standardize character representations.

Example

Input

```
क + ़
```

Output

```
क़
```

---

### Invisible Character Removal

Removes hidden Unicode characters commonly introduced by:

- PDFs
- Microsoft Word
- Web pages
- Copy-paste operations

Characters removed include:

- Zero Width Space
- Zero Width Joiner
- Zero Width Non Joiner
- Byte Order Mark
- Non-breaking Space

---

### Whitespace Normalization

Transforms:

```
नमस्ते      दुनिया
```

into

```
नमस्ते दुनिया
```

Also normalizes:

- Tabs
- Multiple blank lines

---

### Quote Normalization

Converts various Unicode quote styles into standard ASCII quotes.

Example

```
“नमस्ते”
```

↓

```
"नमस्ते"
```

---

### Punctuation Normalization

Collapses repeated punctuation.

Examples

```
नमस्ते।।।
```

↓

```
नमस्ते।
```

```
वाह!!!!!
```

↓

```
वाह!
```

```
क्या???
```

↓

```
क्या?
```

---

## Public API

```python
from preprocessing import HindiNormalizer

normalizer = HindiNormalizer()

clean_text = normalizer.normalize(text)
```

---

## Exceptions

The module raises custom exceptions.

| Exception | Description |
|-----------|-------------|
| InvalidInputTypeError | Input is not a string |
| EmptyTextError | Input string is empty |

---

## Time Complexity

Each normalization step scans the input once.

Overall complexity:

```
O(n)
```

where **n** is the number of characters.

---

## Design Principles

- Single Responsibility Principle
- Small reusable functions
- No modification of sentence meaning
- Extensible pipeline
- Independent unit-testable stages

---

## Example

Input

```
नमस्ते।।।

     मेरा      नाम    शुभम     है।।।

मैं      पुणे     में     रहता हूँ!!!!
```

Output

```
नमस्ते.

मेरा नाम शुभम है.

मैं पुणे में रहता हूँ!
```

---

## Next Module

The output of this module becomes the input for the **Chunk Generation Module**, which assumes:

- clean Unicode
- normalized whitespace
- normalized punctuation
- no hidden Unicode characters