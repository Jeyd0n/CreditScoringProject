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


def export_train(dataframe: pd.DataFrame, connection):
    """Экспорт тренировочного набора данных в локальную базу данных"""

    logger.info('Начало загрузки тренировочного набора данных в базу данных...')
    dataframe.to_sql(
        'train_data',
        con=connection,
        if_exists='replace',
        index=False
    )
    logger.info('Тренировочный набор данных успешно загружен в базу данных')


def export_test(dataframe: pd.DataFrame, connection):
    """Экспорт тестового набора данных в локальную базу данных"""

    logger.info('Начало загрузки тестового набора данных в базу данных...')
    dataframe.to_sql(
        'test_data',
        con=connection,
        if_exists='replace',
        index=False
    )
    logger.info('Тестовый набор данных успешно загружен в базу данных')


def main():
    engine = create_engine(os.getenv('POSTGRESQL_KEY'))
    connection = engine.connect()

    export_train(
        dataframe=pd.read_csv('data/raw/train.csv', low_memory=False),
        connection=connection
    )
    export_test(
        dataframe=pd.read_csv('data/raw/test.csv', low_memory=False),
        connection=connection
    )

    connection.close()
    logger.info('Все данные успешно загружены. Соединение с базой данных закрыто')


if __name__ == '__main__':
    main()
