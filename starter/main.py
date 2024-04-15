from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from ml.data import process_data
from ml.model import inference, load_model
from typing import List, Union
import json


# Define Pydantic model for POST request body
class InferenceRequest(BaseModel):
    age: List[int]
    workclass: List[str]
    fnlgt: List[int]
    education: List[str]
    education_num: List[int] = Field(..., alias="education-num")
    marital_status: List[str] = Field(..., alias="marital-status")
    occupation: List[str]
    relationship: List[str]
    race: List[str]
    sex: List[str]
    capital_gain: List[int] = Field(..., alias="capital-gain")
    capital_loss: List[int] = Field(..., alias="capital-loss")
    hours_per_week: List[int] = Field(..., alias="hours-per-week")
    native_country: List[str] = Field(..., alias="native-country")
    salary: List[str]


class TaggedItem(BaseModel):
    name: str
    tags: Union[str, List[str]]
    item_id: int


# Initialize FastAPI app
app = FastAPI()


# GET request handler
@app.get("/")
async def root():
    return {"message": "Welcome to Zeno's ML model API!"}


# POST request handler for model inference
@app.post("/infer/")
async def inference_endpoint(request: InferenceRequest):
    df = pd.DataFrame(json.loads(request.json()))

    encoder = load_model("../model/encoder.pkl")
    lb = load_model("../model/label_binarizer.pkl")
    X_test, y_test, e_, lb_ = process_data(
        df,
        categorical_features=[
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ],
        label="salary",  # Label is not needed for inference
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Load trained model (you may want to load from a saved file)
    trained_model = load_model("../model/model.pkl")

    # Perform inference
    y_preds = inference(trained_model, X_test)

    return {"predictions": str(y_preds)}
