import joblib
from typing import List, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class Model:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def _load_model(self) -> CatBoostClassifier:
        with open(self.model_path, "rb") as f:
            self.model = joblib.load(f)

    def infer(self, input_data: Union[pd.DataFrame, np.array, List]):
        self._load_model()
        output = self.model.predict(input_data)
        return output
