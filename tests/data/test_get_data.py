import sys

sys.path.append('..')

import pandas as pd
from modules.data.get_data import get_data


def test_get_data(is_train=False):
    data = get_data(is_train)

    assert data.shape[0] == 100_000
    assert data.shape[1] == 28
