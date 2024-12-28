import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.preprocessing import StandardScaler

def calculate_feautres(
    input_data: Union[Dict, pd.DataFrame, np.array, List],
    scaler: StandardScaler,
    col_names: List[str] = None):

    try:
        df = pd.DataFrame.from_dict(input_data, orient='index').T
        df = df[col_names]
    except Exception as e:
        raise ValueError(
            f"Incorrect value supplied for preprocessing: input:{input_data}, error: {e}"
        )

    return scale_features(df, scaler).to_dict(orient='records')

def scale_features(data: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    if not hasattr(scaler, "transform"):
        raise ValueError("Scaler must be a fitted StandardScaler instance.")
    
    scaled_data = scaler.transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)
