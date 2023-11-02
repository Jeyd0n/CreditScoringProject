import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def get_data(is_train=False) -> pd.DataFrame: 
    """Импорт сырого датасета из базы данных"""

    engine = create_engine(os.getenv('POSTGRESQL_KEY'))
    
    with engine.connect() as connection:
        logger.info(f'Начало экспорта набора данных, с параметром is_train={is_train}')

        if is_train:
            dataset = pd.read_sql(
                '''
                SELECT *
                FROM train_data
                ''', 
                con=connection
            )
            logger.info('Тренировочный набор данных успешно экспортирован')

        else:
            dataset = pd.read_sql(
                '''
                SELECT *
                FROM test_data
                ''', 
                con=connection
            )
            logger.info('Тестовый набор данных успешно экспортирован')

    logger.info(dataset.head(5))
    return dataset
