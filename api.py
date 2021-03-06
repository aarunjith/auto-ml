from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from utils.h2o_utils import validate
from utils.utils import convert_label_map
import pandas as pd
from loguru import logger
import json

with open('config/model_config.json', 'r') as f:
    config = json.load(f)
with open('config/data_constants.json', 'r') as f:
    data_constants = json.load(f)

columns = [col for col in config['columns'] if col != config['label']]
model_path = config['model_path']
id_column = config['index']
features = config['features']
label_map = config['label_map']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/validate/")
async def validate_performance(input: Request):
    input = await input.json()
    data = input["data"]
    data = pd.DataFrame(data)
    for col in data.columns:
        if col in columns:
            try:
                if data[col].isna().sum() > 0:
                    imputes = (data[col].isna()).astype(int)
                    data[col] = data[col].fillna(data_constants[col])
                    data[col+'_imputed'] = imputes
                else:
                    data[col+'_imputed'] = 0
            except:
                logger.info(f'Constants for column {col} not found')
                continue
    data = data[features]
    if id_column in data.columns:
        predictions = validate(data, model_path, id_column)
    else:
        predictions = validate(data, model_path)
    predictions = convert_label_map(predictions.to_dict(orient="records"), label_map=label_map)
    return {"predictions": predictions}
