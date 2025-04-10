import os
import numpy as np

DATA_FOLDER_PATH = os.path.abspath(os.path.join('..', '..', 'data', 'processed'))
EXTRACT_LABEL = {
    'ravdess': lambda file_name: int(file_name.split("-")[2]),
    'miami_deception': lambda file_name: 1 if file_name.strip('.npy')[-1] == 'L' else 0,
    'silesian_deception': lambda file_name: 0 if file_name.strip('.npy').split("_")[-1] in ["1", "2", "9"] else 1,
    'nemo_smile': lambda file_name: 1 if "deliberate_smile" in file_name else 0,
}


def load_data(dataset_name: str) -> tuple[np.array, np.array]:
    """
    Load the selected dataset from processed data and return full dataset
    and labels.
    """
    if dataset_name not in os.listdir(DATA_FOLDER_PATH):
        raise FileNotFoundError(f"The dataset {dataset_name} could not be found.")

    data_path = os.path.join(DATA_FOLDER_PATH, dataset_name)

    all_data = []
    all_labels = []

    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(data_path, file), allow_pickle=True)
            data = np.array(data, dtype=np.float32)
            all_data.append(data)
            label = EXTRACT_LABEL[dataset_name](file_name=file)
            all_labels.append(label)

    return np.array(all_data, dtype=object), np.array(all_labels)
