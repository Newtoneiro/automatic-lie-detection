import cv2
import numpy as np
import mediapipe as mp
import supervision as sv
from abc import ABC, abstractmethod


class FrameProcessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return the processed frame.

        Args:
            frame (np.ndarray): The input frame to process.

        Returns:
            (np.ndarray): The processed frame.
        """
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
        resolution_wh = (frame.shape[1], frame.shape[0])
        processed_frame = self._model.process(frame)
        key_points = sv.KeyPoints.from_mediapipe(processed_frame, resolution_wh)

        return self._annotator.annotate(frame, key_points)


class GoogleFaceLandmarkDetectionProcessor(FrameProcessor):
    """
    A frame processor that detects facial landmarks using Google's Face Landmark model
    and annotates them with vertices.
    """

    DEFAULT_COLOR = sv.Color.WHITE
    DEFAULT_RADIUS = 3

    def __init__(
        self,
        model_path: str,
        vertex_color: sv.Color = DEFAULT_COLOR,
        vertex_radius: int = DEFAULT_RADIUS,
    ):
        self._model = self._init_model(model_path)
        self._annotator = sv.VertexAnnotator(color=vertex_color, radius=vertex_radius)

    def _init_model(self, model_path: str) -> mp.tasks.vision.FaceLandmarker:  # type: ignore
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )

        return FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        resolution_wh = (frame.shape[1], frame.shape[0])
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._model.detect(image)
        key_points = sv.KeyPoints.from_mediapipe(result, resolution_wh)

        return self._annotator.annotate(frame, key_points)
