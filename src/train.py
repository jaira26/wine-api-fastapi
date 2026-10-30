import json
import sys
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_NAMES,
    METRICS_PATH,
    MIN_ACCURACY,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
)
from src.data import load_data, split_data


def build_pipeline():
    """Scaling and the classifier live in one pipeline so the API never
    receives unscaled inputs by mistake."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            ),
        ]
    )


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "test_samples": int(len(y_test)),
    }


def main():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

    bundle = {
        "model": pipeline,
        "feature_names": FEATURE_NAMES,
        "class_names": class_names,
        "metrics": metrics,
        "model_type": "RandomForestClassifier",
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics: {metrics}")

    # Quality gate: a non-zero exit fails the CI run if the model regresses.
    if metrics["accuracy"] < MIN_ACCURACY:
        print(f"Accuracy {metrics['accuracy']} is below threshold {MIN_ACCURACY}")
        sys.exit(1)

    return metrics


if __name__ == "__main__":
    main()
