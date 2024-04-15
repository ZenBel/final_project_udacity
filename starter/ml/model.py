from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    return LogisticRegression().fit(X_train, y_train)


def compute_metrics_by_slice(data, feature, cat_features, label, model, encoder, lb):
    unique_values = data[feature].unique()
    with open("../data/slice_output.txt", "w") as f:
        for value in unique_values:
            slice_data = data[data[feature] == value]
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            y_preds = inference(model, X_slice)
            p, r, f1 = compute_model_metrics(y_slice, y_preds)
            f.write(f"Performance metrics for {feature} = {value}:\n")
            f.write(f"Precision: {p}, Recall: {r}, F-1: {f1}\n\n")


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def load_model(filepath):
    """Load a persisted model from a file.

    Args:
        filepath (str): path to the persisted model file

    Raises:
        Exception: "Model file must be a .pkl file"
    Returns:
        _type_ : trained model or encoder or label binarizer
    """
    if filepath.endswith(".pkl"):
        return pickle.load(open(filepath, "rb"))
    else:
        raise Exception("Model file must be a .pkl file")


def save_model(model, filepath):
    """Save a trained model.

    Args:
        model (): trained model or encoder or label binarizer
        filepath (str): path where to save the model
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
