"""
Constants used by the Hindi text normalization pipeline.
"""

import re
import unicodedata

# -----------------------------------------
# Unicode
# -----------------------------------------

UNICODE_FORM = "NFC"

# -----------------------------------------
# Whitespace
# -----------------------------------------

SPACE = " "

TAB = "\t"

NEWLINE = "\n"

DOUBLE_NEWLINE = "\n\n"

# -----------------------------------------
# Invisible Unicode Characters
# -----------------------------------------

INVISIBLE_CHARACTERS = [
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non Joiner
    "\u200d",  # Zero Width Joiner
    "\ufeff",  # BOM
    "\u00a0",  # Non-breaking space
]

# -----------------------------------------
# Hindi Punctuation
# -----------------------------------------

DANDA = "।"

DOUBLE_DANDA = "॥"

# -----------------------------------------
# Quote Mapping
# -----------------------------------------

QUOTE_MAP = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',

    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
}

# -----------------------------------------
# Regex Patterns
# -----------------------------------------

MULTIPLE_SPACES = re.compile(r"[ ]{2,}")

MULTIPLE_NEWLINES = re.compile(r"\n{3,}")

MULTIPLE_TABS = re.compile(r"\t+")

MULTIPLE_DANDA = re.compile(r"।{2,}")

MULTIPLE_EXCLAMATION = re.compile(r"!{2,}")

MULTIPLE_QUESTION = re.compile(r"\?{2,}")

MULTIPLE_COMMA = re.compile(r",{2,}")