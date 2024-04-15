from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference, load_model
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": [19, 37, 81],
                    "workclass": ["Private", "Private", "Federal-gov"],
                    "fnlgt": [212800, 185556, 182898],
                    "education": ["Some-college", "HS-grad", "Some-college"],
                    "education_num": [10, 9, 10],
                    "marital_status": [
                        "Never-married",
                        "Married-civ-spouse",
                        "Married-civ-spouse",
                    ],
                    "occupation": ["Sales", "Craft-repair", "Adm-clerical"],
                    "relationship": ["Own-child", "Husband", "Other-relative"],
                    "race": ["White", "White", "Asian"],
                    "sex": ["Female", "Male", "Female"],
                    "capital_gain": [0, 0, 1887],
                    "capital_loss": [0, 1567, 76],
                    "hours_per_week": [40, 35, 40],
                    "native_country": ["United-States", "Italy", "United-States"],
                    "salary": ["<=50K", "<=50K", ">50K"],
                }
            ]
        }
    }


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

    encoder = load_model("./model/encoder.pkl")
    lb = load_model("./model/label_binarizer.pkl")
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
    trained_model = load_model("./model/model.pkl")

    # Perform inference
    y_preds = inference(trained_model, X_test)

    return {"predictions": str(y_preds)}
