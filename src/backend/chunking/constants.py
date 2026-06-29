"""
Hindi chunking constants and punctuation rules.
"""

import re


MAX_CHARS_DEFAULT = 180

HINDI_SENTENCE_BOUNDARIES = ["।", "?", "!"]

PREFERRED_SPLIT_CHARACTERS = [
    "।",
    ",",
    ";",
    ":",
]

PARAGRAPH_SEPARATOR = "\n\n"

WHITESPACE_SPLIT = re.compile(r"\s+")

MULTIPLE_SPACES = re.compile(r"[ ]{2,}")
