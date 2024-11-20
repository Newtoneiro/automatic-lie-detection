from typing import Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod


class FrameProcessor(ABC):
    def forced_out_size(self) -> Optional[Tuple[int, int]]:
        """
        Returns the output size as a tuple of integers (width, height).
        Returns None, if processor doesnt force out size.
        Returns:
            Optional[Tuple[int, int]]: The output size if available, otherwise None.
        """

        return None

    @abstractmethod
    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame and return the processed frame. Returns
        None if something went wrong

        Args:
            frame (np.ndarray): The input frame to process.

        Returns:
            Optional[np.ndarray]: The processed frame if available, otherwise None.
        """
        pass
