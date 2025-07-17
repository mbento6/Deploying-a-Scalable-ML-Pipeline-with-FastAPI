import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Load saved encoder and model
encoder_path = os.path.join("model", "encoder.pkl")
model_path = os.path.join("model", "model.pkl")

encoder = load_model(encoder_path)
model = load_model(model_path)

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# Map 0/1 to human-readable labels
def apply_label(val):
    return "<=50K" if val == 0 else ">50K"

# GET endpoint
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Welcome to the income inference API!"}

# POST endpoint
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

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

    # Process the input data
    data_processed, _, _, _ = process_data(
        X=data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder
    )

    # Make prediction
    _inference = inference(model, data_processed)[0]
    return {"result": apply_label(_inference)}
