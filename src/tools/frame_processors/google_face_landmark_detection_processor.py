from typing import Optional
import cv2
import numpy as np
import mediapipe as mp
import supervision as sv
from tools.frame_processors import FrameProcessor


class GoogleFaceLandmarkDetectionProcessor(FrameProcessor):
    """
    A frame processor that detects facial landmarks using Google's Face Landmark model
    and annotates them with vertices.
    """

    DEFAULT_COLOR = sv.Color.WHITE
    DEFAULT_RADIUS = 1

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
            base_options=BaseOptions(
                model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        return FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        resolution_wh = (frame.shape[1], frame.shape[0])
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._model.detect(image)
        key_points = sv.KeyPoints.from_mediapipe(result, resolution_wh)

        return self._annotator.annotate(frame, key_points)
