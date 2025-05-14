import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.nn.utils.rnn import pad_sequence

TRAIN_REST_RATIO = 0.3
VAL_TEST_RATIO = 0.5
SEED = 123


def preprocess_data(data: np.array, labels: np.array, binarize_labels=True) -> tuple[np.array]:
    """
    Preprocess given sequential data for training and evaluation.
    Split the results between train, val and test set.
    """
    tensor_data = [torch.tensor(d, dtype=torch.float32) for d in data]
    padded_data = pad_sequence(tensor_data, batch_first=True)

    if binarize_labels:
        encoder = LabelBinarizer()
        labels = encoder.fit_transform(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        padded_data, torch.tensor(labels, dtype=torch.float32), test_size=TRAIN_REST_RATIO, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_TEST_RATIO, random_state=SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
