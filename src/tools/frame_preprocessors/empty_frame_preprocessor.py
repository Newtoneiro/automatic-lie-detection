import numpy as np
from typing import Optional
from tools.frame_preprocessors import FramePreprocessor


class EmptyFramePreprocessor(FramePreprocessor):
    """
    A frame preprocessor that does nothing.
    """

    def preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        return frame
