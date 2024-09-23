import mediapipe as mp
import supervision as sv
import numpy as np
import cv2
from tqdm import tqdm


class DataProcessor:
    def __init__(self) -> None:
        self._model = mp.solutions.face_mesh.FaceMesh()
        self._edge_annotator = sv.EdgeAnnotator(color=sv.Color.WHITE, thickness=1)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame by detecting facial landmarks and annotating
        edges.

        Args:
            frame (np.ndarray): The input frame to process.

        Returns:
            (np.ndarray): The processed frame.
        """
        resolution_wh = (frame.shape[1], frame.shape[0])
        processed_frame = self._model.process(frame)
        key_points = sv.KeyPoints.from_mediapipe(processed_frame, resolution_wh)
        return self._edge_annotator.annotate(frame, key_points)

    def process_video(
        self,
        source_path: str,
        target_path: str = None,
    ):
        """
        Process a video by detecting facial landmarks and annotating edges.
        If no target path is provided, the processed video will be displayed
        instead.

        Args:
            source_path (str): The path to the source video.
            target_path (str): The path to the target video
        """
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)

        if target_path:
            with sv.VideoSink(target_path, video_info) as sink:
                for frame in tqdm(frame_generator, total=video_info.total_frames):
                    annotated_frame = self._process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                annotated_frame = self._process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
