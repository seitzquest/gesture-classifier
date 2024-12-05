import io
import logging
import os
import pickle
import re

import numpy as np
import rarfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set logging level to warning to avoid unnecessary output
logging.getLogger().setLevel(logging.WARNING)

DATA_DIR = "./data/"
NUM_CLASSES = 8 # Number of gestures in the dataset
NUM_FEATURES = 945 # Number of features of the longest time series in the dataset
NUM_DIMENSIONS = 3 # Number of dimensions of the time series

def load_txt_files_from_rar_into_arrays(rar_path):
    """
    Loads all .txt files from a rar archive into numpy arrays.

    Args:
        rar_path (string): Path to the rar archive

    Returns:
        tuple[list[np.ndarray],list[int]]: A tuple containing a list of time series and a list of corresponding gesture labels
    """
    time_series, gesture_labels = [], []

    # The .txt files are named as [somePrefix]$gestureIndex-$repeatIndex.txt, where $gestureIndex is the index of the gesture as in the 8-gesture vocabulary
    file_regex = r".*(\d+)-\d+\.txt$"

    with rarfile.RarFile(rar_path) as rf:
        # Iterate through each file in the rar archive
        for rinfo in rf.infolist():
            match = re.match(file_regex, rinfo.filename)
            if match:
                # Extract the gesture and repeat index as integers
                gestureIndex = int(match.groups()[0])

                with rf.open(rinfo) as file:
                    # Convert file into a text stream
                    text_stream = io.TextIOWrapper(file)
                    data = np.loadtxt(text_stream)
                    time_series.append(data)
                    gesture_labels.append(gestureIndex)
            else:
                logging.info(f"File {rinfo.filename} does not match the expected naming scheme")
    return time_series, gesture_labels

def validate_time_series(ts, label):
    """Validates a num

    Args:
        ts (np.ndarray): Time series to validate
        label (int): Label to validate

    Returns:
        bool: True if the time series is valid, False otherwise
    """

    if len(ts.shape) == 1:
        logging.warning(f"Valid gestures consist of at least two points")
        return False

    if not (len(ts.shape) == 2 and ts.shape[1] == NUM_DIMENSIONS):
        logging.warning(f"Time series has wrong shape {ts.shape}")
        return False
    
    if not all([type(x) is np.float64 for x in ts.flatten()]):
        logging.warning("Time series contains non-float values")
        return False

    if not type(label) is int:
        logging.warning("Label is not an integer")
        return False

    if not (label >= 1 and label <= NUM_CLASSES):
        logging.warning(f"Label is not in the range [1, {NUM_CLASSES}]")
        return False

    return True


def load_data(test_size=0.2, data_path=DATA_DIR):
    """Loads and validates the data from the extracted files of the dataset into numpy arrays, representing the training and test data 

    Args:
        test_size (float, optional): Ratio of test set in train-test-split. Defaults to 0.2.
        data_path (string, optional): Path of the dataset. Defaults to DATA_DIR.

    Returns:
        if test_size is in (0,1):
            tuple[list[np.ndarray], list[int]]: A tuple containing the training and test dataset as numpy arrays
        else:
            tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]: A tuple containing the dataset as numpy arrays (no splitting if test_size is not in (0,1))
    """
    X, y = [], []

    for data_subset in os.listdir(data_path):
        # Skip non-rar files
        if data_subset.endswith('.rar'):
            rar_path = os.path.join(data_path, data_subset)
            time_series, gesture_labels = load_txt_files_from_rar_into_arrays(rar_path)

            if not time_series or not gesture_labels:
                logging.warning("Time series is empty")
                continue
            
            if not len(time_series) == len(gesture_labels):
                logging.warning("Time series and labels have different length")
                continue

            for ts, label in zip(time_series, gesture_labels):
                if validate_time_series(ts, label):
                    X.append(ts)
                    y.append(label)

    if test_size > 0 and test_size < 1:
        return train_test_split(X, y, test_size=test_size)
    else:
        return X, y


def flatten_and_pad(X, num_features_padding=NUM_FEATURES):
    """Flattens the 3D arrays into 2D arrays and pads them with zeros to make them all the same length

    Args:
        X (list[np.ndarray]): List of 3D arrays to be flattened and padded
        num_features_padding (int, optional): Number of features for zero-padding. Defaults to NUM_FEATURES.

    Returns:
        list[np.ndarray]: List of flattened and padded 2D arrays
    """

    X = [x.flatten() for x in X]

    # Trim the time series to the maximum length and pad with zeros
    X = [x[:num_features_padding] for x in X]
    X = [np.pad(x, (0, num_features_padding - len(x)), 'constant') for x in X]
    return X

def data_preprocessing(X_train, X_test, num_features_padding=NUM_FEATURES, normalize=True):
    """Preprocesses the data by flattening and padding the 3D arrays and normalizing the data

    Args:
        X_train (list[np.ndarray]): Training dataset of 3D arrays
        X_test (list[np.ndarray]): Test dataset of 3D arrays
        num_features_padding (int, optional): Number of features for zero-padding. Defaults to NUM_FEATURES.
        normalize (bool, optional): If True the data is normalized using StandardScaler. Defaults to True.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing the training and test dataset after preprocessing.
    """
    # Our classifiers expect 2D arrays of same length (NUM_FEATURES) as input for fitting
    X_train = flatten_and_pad(X_train, num_features_padding)
    X_test = flatten_and_pad(X_test, num_features_padding)

    # Normalize the data
    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test


def get_dataset(test_size=0.2, reload_data=False):
    """Loads the dataset from the extracted files of the dataset into numpy arrays, representing the training and test data.
       Uses the stored dataset of the same test_size if it exists and reload_data is False.

    Args:
        test_size (float, optional): Ratio for train-test split. Defaults to 0.2. Expects a value in (0,1).
        reload_data (bool, optional): If True the data is reloaded even if there already exists a previously stored dataset of same test_size. Defaults to False.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]: A tuple containing the dataset as numpy arrays
    """
    if not (test_size > 0 and test_size < 1):
        logging.warning("test_size must be in (0,1)")

    if not reload_data and os.path.exists(f"dataset_test_size={test_size:.0%}.pkl"):
        X_train, X_test, y_train, y_test = pickle.load(open(f"dataset_test_size={test_size:.0%}.pkl", "rb"))
    else:
        X_train, X_test, y_train, y_test = load_data(test_size=test_size)
        X_train, X_test = data_preprocessing(X_train, X_test)
        pickle.dump((X_train, X_test, y_train, y_test), open(f"dataset_test_size={test_size:.0%}.pkl", "wb"))
    
    return X_train, X_test, y_train, y_test 