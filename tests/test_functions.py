import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter.ml.model import compute_model_metrics, inference
from starter.ml.data import process_data


def test_compute_model_metrics_type():
    """Test whether compute_model_metrics returns the expected type of metrics."""
    # Dummy data
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0]

    # Call the function
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    # Check types
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)


def test_inference_type():
    """Test whether inference returns the expected type of predictions."""
    # Dummy data
    model = LogisticRegression()
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([1, 0, 1])

    model.fit(X, y)

    # Call the function
    predictions = inference(model, X)

    # Check type
    assert isinstance(predictions, np.ndarray)


def test_process_data_output_type():
    sample_data = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "label": [0, 1, 0]}
    )

    X, y, encoder, lb = process_data(
        sample_data, categorical_features=["feature2"], label="label", training=True
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
