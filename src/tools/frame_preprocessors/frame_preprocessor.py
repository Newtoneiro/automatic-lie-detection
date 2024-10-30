import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class FramePreprocessor(ABC):
    def forced_out_size(self) -> Optional[Tuple[int, int]]:
        """
        Returns the output size as a tuple of integers (width, height).
        Returns None, if processor doesnt force out size.
        Returns:
            Optional[Tuple[int, int]]: The output size if available, otherwise None.
        """

        return None

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame and return the processed frame, or None if the
        frame should be skipped.

        Args:
            frame (np.ndarray): The input frame to process.

        Returns:
            (np.ndarray): The processed frame.
        """
        pass
