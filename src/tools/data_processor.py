import numpy as np
import cv2
import supervision as sv
from tools.frame_processors import FrameProcessor
from tqdm import tqdm


class DataProcessor:
    def __init__(self, frame_processor: FrameProcessor) -> None:
        self._frame_processor = frame_processor

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame by detecting facial landmarks and annotating
        edges.

        Args:
            frame (np.ndarray): The input frame to process.

        Returns:
            (np.ndarray): The processed frame.
        """
        return self._frame_processor.process(frame)

    def _process_and_save_video(self, source_path: str, target_path: str) -> None:
        """
        Process a video by detecting facial landmarks and annotating edges.

        Args:
            source_path (str): The path to the source video.
            target_path (str): The path to the target video.
        """
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)

        with sv.VideoSink(target_path, video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                annotated_frame = self._process_frame(frame)
                sink.write_frame(annotated_frame)

    def _process_and_display_video(self, source_path: str) -> None:
        """
        Process a video by detecting facial landmarks and annotating edges.

        Args:
            source_path (str): The path to the source video.
        """
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)

        for frame in tqdm(frame_generator, total=video_info.total_frames):
            annotated_frame = self._process_frame(frame)
            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

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
        if target_path:
            self._process_and_save_video(source_path, target_path)
        else:
            self._process_and_display_video(source_path)
