from typing import NamedTuple, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
import supervision as sv
from tools.frame_processors import FrameProcessor


class SupervisionVertexProcessorWithLandmarkFrontalization(FrameProcessor):
    """
    A frame processor that detects facial landmarks using Ultralytics FaceMesh
    and annotates them with vertices. Frontalizes the output cloud of points.
    """

    DEFAULT_COLOR = sv.Color.WHITE
    DEFAULT_RADIUS = 1
    DEFAULT_OUT_SIZE = (500, 500)
    DEFAULT_OUT_LANDMARK_RADIUS = 2
    DEFAULT_OUT_LANDMARK_COLOR = (255, 255, 255)
    DEFAULT_OUT_LANDMARK_THICKNESS = -1

    def __init__(
        self,
        reference_points_path: str,
        vertex_color: sv.Color = DEFAULT_COLOR,
        vertex_radius: int = DEFAULT_RADIUS,
        out_size: Tuple[int, int] = DEFAULT_OUT_SIZE,
        out_landmark_radius: int = DEFAULT_OUT_LANDMARK_RADIUS,
        out_landmark_color: Tuple[int, int, int] = DEFAULT_OUT_LANDMARK_COLOR,
        out_landmark_thickness: int = DEFAULT_OUT_LANDMARK_THICKNESS,
    ):
        self._reference_points = np.load(reference_points_path)
        self._out_size = out_size
        self._out_landmark_radius = out_landmark_radius
        self._out_landmark_color = out_landmark_color
        self._out_landmark_thickness = out_landmark_thickness

        self._model = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,  # Enables more precise landmark detection (includes irises)
            max_num_faces=1,
        )
        self._annotator = sv.VertexAnnotator(color=vertex_color, radius=vertex_radius)

    def forced_out_size(self) -> Optional[Tuple[int, int]]:
        return self._out_size

    def _get_xyz_from_processed_frame(self, processed_frame: NamedTuple) -> Optional[np.ndarray]:
        """
        Extracts the x, y, and z coordinates from the processed frame's face landmarks.
        Args:
            processed_frame (NamedTuple): A named tuple containing the processed frame data,
                                          which includes multi_face_landmarks.
        Returns:
            np.ndarray: A numpy array containing the x, y, and z coordinates of the face landmarks.
        """
        if not processed_frame.multi_face_landmarks:
            return None

        face_landmarks = [
            face_landmark.landmark
            for face_landmark in processed_frame.multi_face_landmarks
        ]

        xyz = []
        for face_landmark in face_landmarks:
            prediction_xyz = []
            for landmark in face_landmark:
                prediction_xyz.append(
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]
                )

            xyz.append(prediction_xyz)

        return np.array(xyz)

    def _get_xy_from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """
        Extracts and scales the x and y coordinates from a given array of xyz coordinates.
        Args:
            xyz (np.ndarray): A numpy array of shape (n, 3) where each row represents a point in 3D space (x, y, z).
        Returns:
            np.ndarray: A numpy array of shape (n, 2) containing the x and y coordinates of the input points.
        """

        xy = [[x, y] for (x, y, _) in xyz]

        return np.array(xy)

    def _procrustes_analysis(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Perform Procrustes analysis to align X to Y.

        Args:
            X (np.ndarray): A numpy array of shape (N, 3) representing the source points.
            Y (np.ndarray): A numpy array of shape (N, 3) representing the target points.

        Returns:
            np.ndarray: A numpy array of shape (N, 3) representing the aligned source points.
        """
        # Subtract the centroid (mean) of the points
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)

        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        # Normalize the points (scaling to unit variance)
        X_norm = np.linalg.norm(X_centered)
        Y_norm = np.linalg.norm(Y_centered)

        X_centered /= X_norm
        Y_centered /= Y_norm

        # Compute the optimal rotation matrix using SVD
        U, _, Vt = np.linalg.svd(np.dot(X_centered.T, Y_centered))
        R = np.dot(U, Vt)

        # Apply the rotation matrix to X
        X_aligned = np.dot(X_centered, R)

        # Rescale and shift X_aligned back to original scale and position
        X_aligned = X_aligned * Y_norm + Y_mean

        return X_aligned

    def _make_face_mesh(self, frame: np.ndarray) -> np.ndarray:
        """
        Generates a face mesh visualization from the given frame.
        This method creates an empty display frame and draws circles on it
        based on the points provided in the input frame. Each point is scaled
        according to the output size and drawn with specified radius, color,
        and thickness.
        Args:
            frame (np.ndarray): An array of points representing facial landmarks.
                    Each point should be a tuple or list with two
                    elements corresponding to the x and y coordinates.
        Returns:
            np.ndarray: An image (numpy array) with the face mesh drawn on it.
        """

        display_frame = np.zeros(
            (self._out_size[0], self._out_size[1], 3), dtype=np.uint8
        )

        for point in frame:
            cv2.circle(
                display_frame,
                (int(point[0] * self._out_size[0]), int(point[1] * self._out_size[1])),
                self._out_landmark_radius,
                self._out_landmark_color,
                self._out_landmark_thickness,
            )

        return display_frame

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        processed_frame = self._model.process(frame)
        image_to_frontalize_xyz = self._get_xyz_from_processed_frame(processed_frame)
        if image_to_frontalize_xyz is None:
            return None

        frontalized_keypoints = self._procrustes_analysis(
            image_to_frontalize_xyz[0], self._reference_points[0]
        )
        frontalized_keypoints = self._get_xy_from_xyz(frontalized_keypoints)

        return self._make_face_mesh(frontalized_keypoints)
