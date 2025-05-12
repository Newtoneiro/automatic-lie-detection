import numpy as np

EYE_LANDMARKS = {
    "left": {
        "outer": 33,
        "inner": 133,
        "top": 159,
        "bottom": 145,
        "iris_center": 468
    },
    "right": {
        "outer": 263,
        "inner": 362,
        "top": 386,
        "bottom": 374,
        "iris_center": 473
    }
}

MIN_EYE_SIZE = 1e-6


def convert_landmarks_to_eye_movements(landmarks_data):
    n_samples = landmarks_data.shape[0]
    eye_movements = []
    blinking = []

    for sample_idx in range(n_samples):
        sample_landmarks = landmarks_data[sample_idx]
        n_frames = sample_landmarks.shape[0]

        sample_eye_movements = np.zeros((n_frames, 2))
        blinks_per_frame = np.zeros((n_frames))

        for frame_idx in range(n_frames):
            frame = sample_landmarks[frame_idx]
            is_blinking = True
            total_dx = 0
            total_dy = 0

            for eye in ["left", "right"]:
                outer = frame[EYE_LANDMARKS[eye]["outer"]]
                inner = frame[EYE_LANDMARKS[eye]["inner"]]
                top = frame[EYE_LANDMARKS[eye]["top"]]
                bottom = frame[EYE_LANDMARKS[eye]["bottom"]]
                iris = frame[EYE_LANDMARKS[eye]["iris_center"]]

                eye_center = (outer + inner) / 2
                eye_width = np.linalg.norm(outer - inner)
                eye_height = np.linalg.norm(top - bottom)

                eye_width = max(eye_width, MIN_EYE_SIZE)
                eye_height = max(eye_height, MIN_EYE_SIZE)

                total_dx += (iris[0] - eye_center[0]) / (eye_width / 2)
                total_dy += (iris[1] - eye_center[1]) / (eye_height / 2)

                # detect blinking
                eye_opening = np.linalg.norm(top - bottom)
                eye_ratio = eye_opening / eye_width
                is_blinking = is_blinking and eye_ratio < 0.2

            avg_dx = total_dx / 2
            avg_dy = total_dy / 2
            sample_eye_movements[frame_idx] = [avg_dx, avg_dy]
            blinks_per_frame[frame_idx] = 1 if is_blinking else 0

        eye_movements.append(sample_eye_movements)
        blinking.append(blinks_per_frame)

    return np.array(eye_movements, dtype=object), np.array(blinking, dtype=object)
