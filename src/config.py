from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_ACCURACY = 0.90

# Order must match the column order of sklearn's load_wine() feature matrix.
FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]
