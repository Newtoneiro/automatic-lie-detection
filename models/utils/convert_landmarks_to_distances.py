import numpy as np


def get_distances_from_reference(data, reference_points):
    differences = data - reference_points[:, np.newaxis, :]
    squared_differences = differences**2
    sum_squared_differences = np.sum(squared_differences, axis=2)
    distances = np.sqrt(sum_squared_differences)

    return distances


def normalize_distances(distances):
    first_distances = distances[0, :]
    # Create a mask to avoid division by zero
    mask = (first_distances != 0)
    normalized_distances = np.zeros_like(distances)

    normalized_distances[:, mask] = distances[:, mask] / first_distances[mask][np.newaxis, :]

    return normalized_distances


def convert_landmarks_to_distances(landmarks_data, landmark_indexes, reference_point_idx, normalize=False):
    distances_data = []

    for sequence in landmarks_data:
        selected_data = sequence[:, landmark_indexes, :]
        reference_point_data = sequence[:, reference_point_idx, :]

        distances = get_distances_from_reference(selected_data, reference_point_data)
        if normalize:
            distances = normalize_distances(distances)

        distances_data.append(distances)

    return np.array(distances_data, dtype=object)
