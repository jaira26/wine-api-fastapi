from src.schemas import EXAMPLE_SAMPLE


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert len(body["feature_names"]) == 13
    assert body["class_names"] == ["cultivar_0", "cultivar_1", "cultivar_2"]


def test_predict_known_sample(client):
    # First row of the Wine dataset, true label is cultivar_0.
    response = client.post("/predict", json=EXAMPLE_SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] == "cultivar_0"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_negative_value(client):
    response = client.post("/predict", json={**EXAMPLE_SAMPLE, "alcohol": -1})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in EXAMPLE_SAMPLE.items() if k != "proline"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict(client):
    response = client.post("/predict/batch", json={"samples": [EXAMPLE_SAMPLE] * 3})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


def test_batch_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"samples": []})
    assert response.status_code == 422
