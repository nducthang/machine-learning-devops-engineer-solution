from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
import os
from ml.data import process_data
from ml.model import inference
import pandas as pd
import numpy as np

if os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
        
# Initialize API object
app = FastAPI()

# Load model
file_dir = os.path.dirname(__file__)
model_path = os.path.join(file_dir, './model/rf_model.pkl')
encoder_path = os.path.join(file_dir, './model/encoder.pkl')
lb_path = os.path.join(file_dir, './model/lb.pkl')

model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
lb = pickle.load(open(lb_path, 'rb'))


class InputData(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')


@app.get('/')
async def welcome():
    return "Welcome!"


@app.post('/predict')
async def predict(data: InputData):
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
    sample = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}
    input_data = pd.DataFrame.from_dict(sample)
    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    output = inference(model=model, X=X)[0]
    str_out = '<=50K' if output == 0 else '>50K'
    return {"pred": str_out}


if __name__ == '__main__':
    config = uvicorn.config("main:app", host="0.0.0.0",
                            reload=True, port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()
