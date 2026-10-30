from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def load_data():
    """
    Load the Wine dataset.

    Returns:
        X (numpy.ndarray): Feature matrix of shape (178, 13).
        y (numpy.ndarray): Target labels (0, 1, 2), one per cultivar.
        class_names (list[str]): Human-readable class names.
    """
    wine = load_wine()
    class_names = [f"cultivar_{i}" for i in range(len(wine.target_names))]
    return wine.data, wine.target, class_names


def split_data(X, y):
    """
    Split into stratified train and test sets so each cultivar is
    represented proportionally in both.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
