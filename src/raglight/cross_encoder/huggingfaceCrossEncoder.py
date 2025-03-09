from __future__ import annotations
from typing import Any
from typing_extensions import override
from .crossEncoderModel import CrossEncoderModel
from sentence_transformers import CrossEncoder


class HuggingfaceCrossEncoderModel(CrossEncoderModel):
    """
    Concrete implementation of the CrossEncoderModel for HuggingFace models.

    This class provides a specific implementation of the abstract `CrossEncoderModel` for
    loading and using HuggingFace cross encoder.

    Attributes:
        model_name (str): The name of the HuggingFace model to be loaded.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes a HuggingfaceCrossEncoderModel instance.

        Args:
            model_name (str): The name of the HuggingFace model to load.
        """
        super().__init__(model_name)

    @override
    def load(self) -> HuggingfaceCrossEncoderModel:
        """
        Loads the HuggingFace cross encoder model.

        This method overrides the abstract `load` method from the `CrossEncoderModel` class
        and initializes the HuggingFace cross encoder model with the specified `model_name`.

        Returns:
            HuggingfaceCrossEncoderModel: The loaded HuggingFace cross encoder model.
        """
        return CrossEncoder(model_name=self.model_name)
