from src.config import FEATURE_NAMES, MIN_ACCURACY
from src.predict import load_model, predict
from src.schemas import EXAMPLE_SAMPLE


def test_model_meets_accuracy_threshold():
    bundle = load_model()
    assert bundle["metrics"]["accuracy"] >= MIN_ACCURACY


def test_probabilities_sum_to_one():
    bundle = load_model()
    row = [EXAMPLE_SAMPLE[name] for name in FEATURE_NAMES]
    result = predict(bundle, [row])[0]
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
