import sys

sys.path.append('..')


import pytest
from modules.data.get_data import get_data


@pytest.mark.parametrize('is_train', [True, False])
def test_get_data(is_train: bool):
    data = get_data(is_train)

    if is_train:
        assert data.shape[0] == 100_000
        assert data.shape[1] == 28

    else: 
        assert data.shape[0] == 50_000
        assert data.shape[1] == 27
        