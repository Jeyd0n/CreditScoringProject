import sys

sys.path.append('..')


import logging
import pandas as pd
from modules.data.data_transformer import DataProcessing


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def transform_data(dataset: pd.DataFrame, is_train=False) -> pd.DataFrame:
    """Обработка сырого датасета с помощью кастомного класса"""

    if is_train:
        X, y = DataProcessing(
            dataset,
            is_train=is_train
        ).transform()
        logger.info('Преобразование данных прошло успешно')
        logger.info(X.head(5))
        logger.info(y.head(5))

        return X, y
    
    else:
        X = DataProcessing(
            dataset,
            is_train=is_train
        ).transform()
        logger.info('Преобразование данных прошло успешно')
        logger.info(X.head(5))

        return X
