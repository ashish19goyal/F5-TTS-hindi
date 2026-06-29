"""
Hindi-specific smart chunker for normalized text.
"""

from __future__ import annotations

import re
from typing import List

from .chunk import Chunk
from .constants import (
    HINDI_SENTENCE_BOUNDARIES,
    MAX_CHARS_DEFAULT,
    PARAGRAPH_SEPARATOR,
    PREFERRED_SPLIT_CHARACTERS,
    WHITESPACE_SPLIT,
)
from .exceptions import EmptyChunkError, InvalidInputTypeError


class ChunkGenerator:

    def __init__(self, max_chars: int = MAX_CHARS_DEFAULT):
        if not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")

        self.max_chars = max_chars

    # Public API
    def chunk(self, text: str) -> List[Chunk]:
        self._validate(text)

        paragraphs = self._split_paragraphs(text)
        sentences = self._split_sentences(paragraphs)
        merged = self._merge_short_sentences(sentences)
        balanced = self._split_oversized_units(merged)
        return self._build_chunks(balanced)

    # -------------------------------------------------
    def _validate(self, text: str):
        if not isinstance(text, str):
            raise InvalidInputTypeError("Input must be a string.")

        if text.strip() == "":
            raise InvalidInputTypeError(
                "Input text cannot be empty or whitespace-only."
            )

    # -------------------------------------------------
    def _split_paragraphs(self, text: str) -> List[str]:
        return [paragraph.strip() for paragraph in text.split(PARAGRAPH_SEPARATOR) if paragraph.strip()]

    # -------------------------------------------------
    def _split_sentences(self, paragraphs: List[str]) -> List[str]:
        sentences: List[str] = []

        for paragraph in paragraphs:
            start = 0
            for match in re.finditer(r"[{}]".format(re.escape("".join(HINDI_SENTENCE_BOUNDARIES))), paragraph):
                end = match.end()
                sentence = paragraph[start:end].strip()
                if sentence:
                    sentences.append(sentence)
                start = end

            tail = paragraph[start:].strip()
            if tail:
                sentences.append(tail)

        return sentences

    # -------------------------------------------------
    def _merge_short_sentences(self, sentences: List[str]) -> List[str]:
        merged: List[str] = []
        current = ""

        for sentence in sentences:
            if not current:
                current = sentence
                continue

            if len(current) + 1 + len(sentence) <= self.max_chars:
                current = f"{current} {sentence}"
                continue

            merged.append(current)
            current = sentence

        if current:
            merged.append(current)

        return merged

    # -------------------------------------------------
    def _split_oversized_units(self, units: List[str]) -> List[str]:
        result: List[str] = []

        for unit in units:
            if len(unit) <= self.max_chars:
                result.append(unit)
                continue

            next_piece = []
            for part in self._split_by_preferred_boundaries(unit):
                if len(part) > self.max_chars:
                    if next_piece:
                        result.append(" ".join(next_piece).strip())
                        next_piece = []
                    result.extend(self._hard_split(part))
                    continue

                if not next_piece:
                    next_piece = [part]
                    continue

                joined = " ".join(next_piece + [part])
                if len(joined) <= self.max_chars:
                    next_piece.append(part)
                else:
                    result.append(" ".join(next_piece).strip())
                    next_piece = [part]

            if next_piece:
                result.append(" ".join(next_piece).strip())

        return result

    # -------------------------------------------------
    def _split_by_preferred_boundaries(self, text: str) -> List[str]:
        pattern = r"([{}])".format(re.escape("".join(PREFERRED_SPLIT_CHARACTERS)))
        parts = re.split(pattern, text)
        output: List[str] = []

        for index in range(0, len(parts), 2):
            chunk = parts[index].strip()
            punctuation = parts[index + 1] if index + 1 < len(parts) else ""
            if chunk or punctuation:
                output.append((chunk + punctuation).strip())

        return [part for part in output if part]

    # -------------------------------------------------
    def _hard_split(self, text: str) -> List[str]:
        words = WHITESPACE_SPLIT.split(text)
        result: List[str] = []
        current = ""

        for word in words:
            if not current:
                current = word
                continue

            if len(current) + 1 + len(word) <= self.max_chars:
                current = f"{current} {word}"
                continue

            result.append(current)
            current = word

        if current:
            result.append(current)

        return result

    # -------------------------------------------------
    def _build_chunks(self, units: List[str]) -> List[Chunk]:
        chunks: List[Chunk] = []

        for order, unit in enumerate(units, start=1):
            if not unit.strip():
                raise EmptyChunkError("Generated chunk is empty.")

            chunks.append(
                Chunk(
                    id=order,
                    order=order,
                    text=unit,
                    length=len(unit),
                )
            )

        return chunks
