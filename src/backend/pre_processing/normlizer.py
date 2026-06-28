"""
Hindi Text Normalizer

Pipeline

Input
    ↓
Unicode Normalization
    ↓
Invisible Character Cleanup
    ↓
Whitespace Cleanup
    ↓
Quote Normalization
    ↓
Punctuation Cleanup
    ↓
Trim
    ↓
Normalized Text
"""

import unicodedata

from .constants import *
from .exceptions import *


class HindiNormalizer:

    def normalize(self, text: str) -> str:

        self._validate(text)

        text = self._unicode_normalize(text)

        text = self._remove_invisible_characters(text)

        text = self._normalize_quotes(text)

        text = self._normalize_whitespace(text)

        text = self._normalize_punctuation(text)

        text = self._trim(text)

        return text

    # -------------------------------------------------

    def _validate(self, text):

        if not isinstance(text, str):
            raise InvalidInputTypeError(
                "Input must be a string."
            )

        if text.strip() == "":
            raise EmptyTextError(
                "Input text cannot be empty."
            )

    # -------------------------------------------------

    def _unicode_normalize(self, text):

        return unicodedata.normalize(
            UNICODE_FORM,
            text
        )

    # -------------------------------------------------

    def _remove_invisible_characters(self, text):

        for char in INVISIBLE_CHARACTERS:
            text = text.replace(char, SPACE)

        return text

    # -------------------------------------------------

    def _normalize_quotes(self, text):

        for src, dst in QUOTE_MAP.items():
            text = text.replace(src, dst)

        return text

    # -------------------------------------------------

    def _normalize_whitespace(self, text):

        text = MULTIPLE_TABS.sub(
            SPACE,
            text
        )

        text = MULTIPLE_SPACES.sub(
            SPACE,
            text
        )

        text = MULTIPLE_NEWLINES.sub(
            DOUBLE_NEWLINE,
            text
        )

        return text

    # -------------------------------------------------

    def _normalize_punctuation(self, text):

        text = MULTIPLE_DANDA.sub(
            DANDA,
            text
        )

        text = MULTIPLE_EXCLAMATION.sub(
            "!",
            text
        )

        text = MULTIPLE_QUESTION.sub(
            "?",
            text
        )

        text = MULTIPLE_COMMA.sub(
            ",",
            text
        )

        return text

    # -------------------------------------------------

    def _trim(self, text):

        return text.strip()