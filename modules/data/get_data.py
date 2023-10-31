import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()

def get_data(is_train=False) -> pd.DataFrame: 
    """Импорт сырого датасета из базы данных"""

    engine = create_engine(os.getenv('POSTGRESQL_KEY'))
    
    with engine.connect() as connection:
        if is_train:
            dataset = pd.read_sql(
                '''
                SELECT *
                FROM train_data
                ''', 
                con=connection
            )

        else:
            dataset = pd.read_sql(
                '''
                SELECT *
                FROM test_data
                ''', 
                con=connection
            )

    return dataset