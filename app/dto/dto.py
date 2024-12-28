from typing import Union
from pydantic import BaseModel


class ModelInput(BaseModel):
    type: Union[int, list]
    amount: Union[float, int, list]
    oldbalanceOrg: Union[float, int, list]
    newbalanceOrig: Union[float, int, list]
    oldbalanceDest: Union[float, int, list]
    newbalanceDest: Union[float, int, list]


class ModelOutput(BaseModel):
    predictions: Union[int, list]
    model_name: str
