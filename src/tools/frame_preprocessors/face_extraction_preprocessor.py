import cv2
import numpy as np
from typing import Optional, Tuple
from deepface import DeepFace
from tools.frame_preprocessors import FramePreprocessor


class FaceExtractionPreprocessor(FramePreprocessor):
    """
    A frame preprocessor that extracts faces from a frame.
    """

    DEFAULT_OUT_SIZE = (200, 200)

    def __init__(
        self, skip_bad_frames: bool = True, output_size: Tuple[int, int] = None
    ) -> None:
        self._skip_bad_frames = skip_bad_frames
        self._out_size = output_size or self.DEFAULT_OUT_SIZE

    def _fit_to_output_size(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize a frame to the output size.
        """
        return cv2.resize(frame, self._out_size)

    def forced_out_size(self) -> Optional[Tuple[int, int]]:
        return self._out_size

    def preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                align=False,
                normalize_face=False,
                color_face="bgr",
            )
            detected_face = face_objs[0]["face"]
            return self._fit_to_output_size(detected_face)
        except ValueError:
            return None if self._skip_bad_frames else frame
