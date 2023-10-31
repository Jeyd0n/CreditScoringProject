import sys

sys.path.append('..')


import pandas as pd
from modules.data.transform_data import transform_data


def test_transform_data(dataset: pd.DataFrame, is_train=False):

    if is_train:
        X, y = transform_data(
            dataset=dataset,
            is_train=is_train
        )

        assert X.shape[0] == 100_000
        assert X.shape[1] == 22
        assert y.shape == 100_000

    else:
        X = transform_data(
            dataset=dataset,
            is_train=is_train
        )

        assert X.shape[0] == 100_000
        assert X.shape[1] == 22