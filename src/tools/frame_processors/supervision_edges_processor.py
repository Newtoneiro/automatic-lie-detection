from typing import Optional
import numpy as np
import mediapipe as mp
import supervision as sv
from tools.frame_processors import FrameProcessor


class SupervisionEdgesProcessor(FrameProcessor):
    """
    A frame processor that detects facial landmarks using Ultralytics FaceMesh
    and annotates them with edges.
    """

    DEFAULT_COLOR = sv.Color.WHITE
    DEFAULT_THICKNESS = 1

    def __init__(
        self, edges_color: sv.Color = DEFAULT_COLOR, thickness: int = DEFAULT_THICKNESS
    ):
        self._model = mp.solutions.face_mesh.FaceMesh()
        self._annotator = sv.EdgeAnnotator(color=edges_color, thickness=thickness)

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        resolution_wh = (frame.shape[1], frame.shape[0])
        processed_frame = self._model.process(frame)
        key_points = sv.KeyPoints.from_mediapipe(processed_frame, resolution_wh)

        return self._annotator.annotate(frame, key_points)
