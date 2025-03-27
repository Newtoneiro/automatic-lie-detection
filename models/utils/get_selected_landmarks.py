import numpy as np


def get_selected_landmarks(data, landmarks_indexes):
    selected_landmarks_data = []

    for sequence in data:
        selected_data = sequence[:, landmarks_indexes, :]
        selected_landmarks_data.append(selected_data)

    return np.array(selected_landmarks_data, dtype=object)
