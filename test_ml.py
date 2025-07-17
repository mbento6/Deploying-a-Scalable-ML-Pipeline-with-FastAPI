import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics

@pytest.fixture
def sample_data():
    """Provides sample training data with at least 3 samples for cross-validation"""
    X = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])
    y = np.array([0, 1, 0, 1, 1, 0])
    return X, y


def test_train_model_returns_model(sample_data):
    """
    Test that train_model returns a trained RandomForestClassifier
    """
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_inference_shape(sample_data):
    """
    Test that inference returns predictions of the correct shape
    """
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape

def test_compute_model_metrics_returns_all(sample_data):
    """
    Test that compute_model_metrics returns precision, recall, and fbeta as floats
    """
    _, y = sample_data
    preds = y  # simulate perfect predictions
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert precision == recall == fbeta == 1.0
