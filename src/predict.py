import joblib
import numpy as np

from src.config import MODEL_PATH


def load_model(path=MODEL_PATH):
    """
    Load the saved model bundle (pipeline plus metadata).

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {path}. Run 'python -m src.train' first."
        )
    return joblib.load(path)


def predict(bundle, rows):
    """
    Predict cultivar classes for one or more samples.

    Args:
        bundle (dict): Model bundle returned by load_model().
        rows (list[list[float]]): Feature rows in FEATURE_NAMES order.

    Returns:
        list[dict]: One result per row with class id, class name,
        confidence, and per-class probabilities.
    """
    model = bundle["model"]
    class_names = bundle["class_names"]
    X = np.asarray(rows, dtype=float)

    probabilities = model.predict_proba(X)
    results = []
    for row_probs in probabilities:
        class_id = int(np.argmax(row_probs))
        results.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": round(float(row_probs[class_id]), 4),
                "probabilities": {
                    name: round(float(p), 4) for name, p in zip(class_names, row_probs)
                },
            }
        )
    return results
