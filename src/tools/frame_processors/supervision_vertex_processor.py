import numpy as np
import mediapipe as mp
import supervision as sv
from tools.frame_processors import FrameProcessor


class SupervisionVertexProcessor(FrameProcessor):
    """
    A frame processor that detects facial landmarks using Ultralytics FaceMesh
    and annotates them with vertices.
    """

    DEFAULT_COLOR = sv.Color.WHITE
    DEFAULT_RADIUS = 1

    def __init__(
        self,
        vertex_color: sv.Color = DEFAULT_COLOR,
        vertex_radius: int = DEFAULT_RADIUS,
    ):
        self._model = mp.solutions.face_mesh.FaceMesh()
        self._annotator = sv.VertexAnnotator(color=vertex_color, radius=vertex_radius)

    def process(self, frame: np.ndarray) -> np.ndarray:
        resolution_wh = (frame.shape[1], frame.shape[0])
        processed_frame = self._model.process(frame)
        key_points = sv.KeyPoints.from_mediapipe(processed_frame, resolution_wh)

        return self._annotator.annotate(frame, key_points)
