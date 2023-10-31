import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()


def export_train(dataframe: pd.DataFrame, connection):
    """Экспорт тренировочного набора данных в локальную базу данных"""

    dataframe.to_sql(
        'train_data',
        con=connection,
        if_exists='replace',
        index=False
    )


def export_test(dataframe: pd.DataFrame, connection):
    """Экспорт тестового набора данных в локальную базу данных"""

    dataframe.to_sql(
        'test_data',
        con=connection,
        if_exists='replace',
        index=False
    )


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


if __name__ == '__main__':
    main()
