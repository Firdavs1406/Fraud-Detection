import warnings
from typing import Dict

import joblib
from app.dto.dto import ModelInput, ModelOutput
from fastapi import FastAPI
from app.model.engine import Model
from app.preprocessing.prepare_data import calculate_feautres

warnings.filterwarnings('ignore')

import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

col_names = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

model_path = 'app/artefacts/catboost_model.joblib'
scaler_path = 'app/artefacts/scaler.joblib'

with open(scaler_path, "rb") as f:
    scaler = joblib.load(f)

@app.get("/health")
async def health_check() -> Dict:
    return {'status': 'success'}


@app.post("/predict")
async def get_predictions(input_data: ModelInput):
    model = Model(model_path)
    preprocessed = calculate_feautres(input_data.dict(), scaler, col_names)
    final_data = [list(v.values()) for v in preprocessed]
    preds = model.infer(final_data)
    model_name = model.model_path.split('/')[-1]
    output = ModelOutput(predictions=preds.tolist(), model_name=model_name)
    return output
