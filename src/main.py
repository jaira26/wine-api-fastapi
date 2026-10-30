from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from src.predict import load_model, predict
from src.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    WineFeatures,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    try:
        app.state.bundle = load_model()
    except FileNotFoundError:
        app.state.bundle = None
    yield
    app.state.bundle = None


app = FastAPI(
    title="Wine Cultivar Classifier API",
    description="Serves a RandomForest pipeline trained on the UCI Wine dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_bundle(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run 'python -m src.train' and restart the server.",
        )
    return bundle


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    loaded = getattr(request.app.state, "bundle", None) is not None
    return HealthResponse(status="healthy" if loaded else "degraded", model_loaded=loaded)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(bundle: dict = Depends(get_bundle)):
    return ModelInfoResponse(
        model_type=bundle["model_type"],
        sklearn_version=bundle["sklearn_version"],
        trained_at=bundle["trained_at"],
        feature_names=bundle["feature_names"],
        class_names=bundle["class_names"],
        metrics=bundle["metrics"],
    )


# Plain def (not async def): model inference is CPU-bound, so FastAPI runs
# these in a threadpool instead of blocking the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict_one(sample: WineFeatures, bundle: dict = Depends(get_bundle)):
    try:
        return predict(bundle, [sample.to_row()])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request_body: BatchRequest, bundle: dict = Depends(get_bundle)):
    try:
        rows = [sample.to_row() for sample in request_body.samples]
        results = predict(bundle, rows)
        return BatchResponse(count=len(results), predictions=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
