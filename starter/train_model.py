# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_model
import pandas as pd

# Load the cleaned data
data = pd.read_csv("../data/census_cleaned.csv")
print("Read census data")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
print("Split data into train and test")

# Proces the test data with the process_data function.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
print("Processed data for model training")

# Save categorical encoder and label binarizer
save_model(encoder, "../model/encoder.pkl")
save_model(lb, "../model/label_binarizer.pkl")
print("Saved encoder and label binarizer to model folder")

X_valid, y_valid, e_, lb_ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
print("Processed test data for model validation")

# Train model.
trained_model = train_model(X_train, y_train)
print("Trained model")

# Compute metrics
y_preds = inference(trained_model, X_valid)
p, r, f = compute_model_metrics(y_valid, y_preds)
print(f"Precision: {p}, Recall: {r}, F-1: {f}")

save_model(trained_model, "../model/model.pkl")
print("Saved trained model to model folder")
