from typing import Optional, Tuple
import numpy as np
import cv2
import supervision as sv
from tools.frame_processors import (
    FrameProcessor,
)
from tools.frame_preprocessors import (
    FramePreprocessor,
    EmptyFramePreprocessor,
)
from tqdm import tqdm


class DataProcessor:
    def __init__(
        self,
        frame_processor: FrameProcessor,
        frame_preprocessor: FramePreprocessor = None,
    ) -> None:
        self._frame_processor = frame_processor
        self._frame_preprocessor = frame_preprocessor or EmptyFramePreprocessor()

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess a single frame by extracting faces.

        Args:
            frame (np.ndarray): The input frame to process.

        Returns:
            Optional(np.ndarray): The processed frame or None if preprocessed frame is bad.
        """
        return self._frame_preprocessor.preprocess(frame)

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

    def _handle_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Handle a single frame by preprocessing and processing it.

        Args:
            frame (np.ndarray): The input frame to handle.

        Returns:
            (np.ndarray): The handled frame.
        """
        preprocessed_frame = self._preprocess_frame(frame)
        if preprocessed_frame is not None:
            return self._process_frame(preprocessed_frame)

        return None

    def _process_and_save_video(self, source_path: str, target_path: str) -> None:
        """
        Process a video by detecting facial landmarks and annotating edges.

        Args:
            source_path (str): The path to the source video.
            target_path (str): The path to the target video.
        """
        video_info = sv.VideoInfo.from_video_path(source_path)
        forced_out_size = self._get_forced_out_size()
        if forced_out_size is not None:
            video_info.width = forced_out_size[0]
            video_info.height = forced_out_size[1]

        frame_generator = sv.get_video_frames_generator(source_path)

        with sv.VideoSink(target_path, video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                annotated_frame = self._handle_frame(frame)
                if annotated_frame is None:
                    continue

                sink.write_frame(annotated_frame)

    def _get_forced_out_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the forced out size forced by processor and preprocessors.
        Processor out size has higher priority.
        Returns None, if processor doesnt force out size.
        Returns:
            Optional[Tuple[int, int]]: The output size if available, otherwise None.
        """
        preprocessor_out_size = self._frame_preprocessor.forced_out_size()
        processor_out_size = self._frame_processor.forced_out_size()

        return (
            processor_out_size
            if processor_out_size is not None
            else preprocessor_out_size
        )

    def _process_and_display_video(self, source_path: str) -> None:
        """
        Process a video by detecting facial landmarks and annotating edges.

        Args:
            source_path (str): The path to the source video.
        """
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)

        for frame in tqdm(frame_generator, total=video_info.total_frames):
            annotated_frame = self._handle_frame(frame)
            if annotated_frame is None:
                continue

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
