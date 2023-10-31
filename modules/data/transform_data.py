import sys

sys.path.append('..')


import pandas as pd
from modules.data.data_transformer import DataProcessing

def transform_data(dataset: pd.DataFrame, is_train=False) -> pd.DataFrame:
    if is_train:
        X, y = DataProcessing(
            dataset,
            is_train=is_train
        ).transform()

        return X, y
    
    else:
        X = DataProcessing(
            dataset,
            is_train=is_train
        ).transform()

        return X

