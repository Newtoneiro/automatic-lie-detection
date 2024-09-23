import numpy as np
import mediapipe as mp
import supervision as sv
from abc import ABC, abstractmethod


class FrameProcessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        pass


class SupervisionVertexProcessor(FrameProcessor):
    """
    A frame processor that detects facial landmarks using Ultralytics FaceMesh
    and annotates them with vertices.
    """

    DEFAULT_COLOR = sv.Color.WHITE
    DEFAULT_RADIUS = 3

    def __init__(
        self,
        vertex_color: sv.Color = DEFAULT_COLOR,
        vertex_radius: int = DEFAULT_RADIUS,
    ):
        self._model = mp.solutions.face_mesh.FaceMesh()
        self._annotator = sv.VertexAnnotator(color=vertex_color, radius=vertex_radius)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame by detecting facial landmarks using Ultralytics
        FaceMesh model.

        Args:
            frame (np.ndarray): The input frame to process.
        """
        resolution_wh = (frame.shape[1], frame.shape[0])
        processed_frame = self._model.process(frame)
        key_points = sv.KeyPoints.from_mediapipe(processed_frame, resolution_wh)

        return self._annotator.annotate(frame, key_points)


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

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame by detecting facial landmarks using Ultralytics
        FaceMesh model.

        Args:
            frame (np.ndarray): The input frame to process.
        """
        resolution_wh = (frame.shape[1], frame.shape[0])
        processed_frame = self._model.process(frame)
        key_points = sv.KeyPoints.from_mediapipe(processed_frame, resolution_wh)

        return self._annotator.annotate(frame, key_points)
