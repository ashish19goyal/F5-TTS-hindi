"""
Custom exceptions for preprocessing module.
"""


class NormalizationError(Exception):
    """
    Base class for all normalization errors.
    """


class EmptyTextError(NormalizationError):
    """
    Raised when input text is empty.
    """


class InvalidInputTypeError(NormalizationError):
    """
    Raised when input is not string.
    """