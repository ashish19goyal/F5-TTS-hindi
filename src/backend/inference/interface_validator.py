"""
Interface validator for inference implementations.
"""

from typing import Type

from .base import BaseInference
from .exceptions import InferenceException


class InferenceInterfaceValidator:
    """
    Validates that an inference implementation meets interface requirements.
    """

    REQUIRED_METHODS = ["infer", "infer_async", "load_resources", "cleanup"]

    @staticmethod
    def validate(inference_engine: BaseInference) -> bool:
        """
        Check that engine has required methods and lifecycle support.

        Returns True if valid, raises InferenceException otherwise.

        Args:
            inference_engine: The inference engine instance to validate

        Returns:
            True if validation passes

        Raises:
            InferenceException: If any required method is missing
        """
        for method in InferenceInterfaceValidator.REQUIRED_METHODS:
            if not hasattr(inference_engine, method):
                raise InferenceException(
                    f"Inference engine missing required method: {method}"
                )

            if not callable(getattr(inference_engine, method)):
                raise InferenceException(
                    f"Inference engine attribute '{method}' is not callable"
                )

        return True

    @staticmethod
    def validate_class(inference_class: Type[BaseInference]) -> bool:
        """
        Validate that a class properly implements BaseInference.

        Args:
            inference_class: The inference class to validate

        Returns:
            True if validation passes

        Raises:
            InferenceException: If validation fails
        """
        if not issubclass(inference_class, BaseInference):
            raise InferenceException(
                f"{inference_class.__name__} must inherit from BaseInference"
            )

        return True
