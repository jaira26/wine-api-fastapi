from pydantic import BaseModel, ConfigDict, Field

from src.config import FEATURE_NAMES

EXAMPLE_SAMPLE = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315": 3.92,
    "proline": 1065.0,
}


class WineFeatures(BaseModel):
    """Chemical analysis of a single wine sample. All values must be non-negative."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_SAMPLE})

    alcohol: float = Field(..., ge=0, description="Alcohol content (%)")
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315: float = Field(..., ge=0, description="OD280/OD315 of diluted wines")
    proline: float = Field(..., ge=0)

    def to_row(self):
        """Return feature values in the exact order the model was trained on."""
        return [getattr(self, name) for name in FEATURE_NAMES]


class BatchRequest(BaseModel):
    samples: list[WineFeatures] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    sklearn_version: str
    trained_at: str
    feature_names: list[str]
    class_names: list[str]
    metrics: dict[str, float]
