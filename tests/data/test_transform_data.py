import sys

sys.path.append('..')


import pytest
import pandas as pd
from modules.data.get_data import get_data
from modules.data.transform_data import transform_data


@pytest.mark.parametrize('is_train', [True, False])
def test_transform_data(is_train: bool):
    dataset = get_data(is_train=is_train)

    if is_train:
        X, y = transform_data(
            dataset=dataset,
            is_train=is_train
        )

        assert X.shape[0] == 100_000
        assert X.shape[1] == 21
        assert y.shape[0] == 100_000

    else:
        X = transform_data(
            dataset=dataset,
            is_train=is_train
        )

        assert X.shape[0] == 50_000
        assert X.shape[1] == 21