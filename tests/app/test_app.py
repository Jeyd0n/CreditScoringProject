import sys

sys.path.append('..')


import pytest
import requests
import pandas as pd
import numpy as np
from modules.data.get_data import get_data
from modules.data.transform_data import transform_data

data = get_data(is_train=True)
X, y = transform_data(
       dataset=data,
       is_train=True
)


@pytest.mark.parametrize('test_batch', [
    X.sample(n=5).to_numpy(),
    X.sample(n=10).to_numpy()
    ])
def test_request_get_predict(test_batch: np.array):
    prediction = requests.post(
        "http://127.0.0.1:3000/get_predict",
        headers={"content-type": "application/json"},
        data=f'{test_batch.tolist()}'
    ).text

    predictions = []
    for value in prediction:
        if value in {'0', '1', '2'}:
            predictions.append(int(value))

    assert len(predictions) == 5 or len(predictions) == 10
    
    for value in predictions:
        assert value in {0, 1, 2}
