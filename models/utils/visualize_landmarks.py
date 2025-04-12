import os
import numpy as np
from matplotlib import pyplot as plt

loaded_key_points = np.load(os.path.join(__file__, '..', '..', '..', 'data', 'reference_points', 'key_points_xyz.npy'))
OFFSET = 0.3


def visualize_landmarks(landmarks_indexes):
    selected_key_points = loaded_key_points[:, landmarks_indexes, :]

    # Plot the key points
    plt.figure(figsize=(6, 6))
    plt.scatter(
        selected_key_points[:, :, 0],
        selected_key_points[:, :, 1],
        c='r',
        marker='.',
        label='Selected Landmarks'
    )

    # Flip Y-axis to match image coordinates
    plt.gca().invert_yaxis()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    OFFSET = 0.3
    plt.xlim(0 - OFFSET, 1 + OFFSET)
    plt.ylim(1 + OFFSET, 0 - OFFSET)
    plt.show()
