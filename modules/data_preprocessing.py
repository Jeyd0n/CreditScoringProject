import pandas as pd
import numpy as np
import scipy.stats as st


class DataProcessing:

    def __init__(self, data: pd.DataFrame, is_train=False):
        self.data = data
        self.is_train = is_train

    def fill_null(self):
        # Функция для полной обработки набора данных

        # Заполнение пропусков в данных
        self.data['Monthly_Inhand_Salary'] = self.data['Monthly_Inhand_Salary'].fillna(np.median(self.data['Monthly_Inhand_Salary'].dropna())).astype(int)
        self.data['Type_of_Loan'] = self.data['Type_of_Loan'].fillna(self.data['Type_of_Loan'].value_counts().index[0])
        self.data['Num_of_Delayed_Payment'] = self.data['Num_of_Delayed_Payment'].fillna(0)

        self.data['Num_Credit_Inquiries'] = self.data['Num_Credit_Inquiries'].fillna(0)
        self.data['Credit_History_Age'] = self.data['Credit_History_Age'].fillna(0)
        self.data['Amount_invested_monthly'] = self.data['Amount_invested_monthly'].fillna(0)
        self.data['Monthly_Balance'] = self.data['Monthly_Balance'].fillna(0)

        # Удаление ненужных колонок
        self.data = self.data.drop(['ID', 'Customer_ID', 'Name', 'Type_of_Loan'], axis=1)

        return self

    def to_type(self):
        # Приведение признаков к нужному типу данных, а так же заполнение пропусков после преобразований

        self.data['Amount_invested_monthly'] = pd.to_numeric(self.data['Amount_invested_monthly'], errors='coerce')
        self.data['Amount_invested_monthly'] = self.data['Amount_invested_monthly'].fillna(st.mode(self.data['Amount_invested_monthly'])[0])

        self.data['Monthly_Balance'] = pd.to_numeric(self.data['Monthly_Balance'], errors='coerce')
        self.data['Monthly_Balance'] = self.data['Monthly_Balance'].fillna(self.data['Monthly_Balance'].median())

        self.data['Annual_Income'] = pd.to_numeric(self.data['Annual_Income'], errors='coerce')
        self.data['Annual_Income'] = self.data['Annual_Income'].fillna(self.data['Annual_Income'].median())

        self.data['Credit_History_Age'] = self.data['Credit_History_Age'].map(lambda x: int(str(x)[0:2]))

        self.data['Num_of_Loan'] = pd.to_numeric(self.data['Num_of_Loan'], errors='coerce')
        self.data['Num_of_Loan'] = self.data['Num_of_Loan'].fillna(self.data['Num_of_Loan'].median())

        self.data["Age"] = pd.to_numeric(self.data["Age"], errors="coerce")
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

        self.data['Outstanding_Debt'] = pd.to_numeric(self.data["Outstanding_Debt"], errors="coerce")
        self.data['Outstanding_Debt'] = self.data['Outstanding_Debt'].fillna(self.data['Outstanding_Debt'].median())

        self.data["Changed_Credit_Limit"] = pd.to_numeric(self.data["Changed_Credit_Limit"], errors="coerce")
        self.data['Changed_Credit_Limit'] = self.data['Changed_Credit_Limit'].fillna(self.data['Changed_Credit_Limit'].median())

        self.data["Num_of_Delayed_Payment"] = pd.to_numeric(self.data["Num_of_Delayed_Payment"], errors="coerce")
        self.data['Num_of_Delayed_Payment'] = self.data['Num_of_Delayed_Payment'].fillna(0)

        return self

    def cat_decode(self):
        # Функция для преобразования категориальных признаков 

        map_table = {
            '...': 0,
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12
        }

        self.data['Month'] = self.data['Month'].replace(map_table)

        self.data['Payment_Behaviour'] = self.data['Payment_Behaviour'].map(lambda x: 'unknown' if x == '!@9#%8' else x)
                
        self.data['Occupation'] = self.data['Occupation'].map(lambda x: 'unknown' if x == '_______' else x)
        occupation_map_table = {k: v for v, k in enumerate(self.data['Occupation'].unique())}
        self.data['Occupation'] = self.data['Occupation'].replace(occupation_map_table)

        self.data['Credit_Mix'] = self.data['Credit_Mix'].map(lambda x: 'NoData' if x == '_' else x)

        # Произведем OHE над некоторыми колонками 
        self.data = pd.concat((self.data.drop('Payment_of_Min_Amount', axis=1), pd.get_dummies(self.data['Payment_of_Min_Amount'], dtype=int)), axis=1)

        self.data = pd.concat((self.data.drop('Payment_Behaviour', axis=1), pd.get_dummies(self.data['Payment_Behaviour'], dtype=int)), axis=1)
        train = train.drop('unknown', axis=1)

        self.data = pd.concat((self.data.drop('Credit_Mix', axis=1), pd.get_dummies(self.data['Credit_Mix'], dtype=int)), axis=1)
        train = train.drop('NoData', axis=1)

        return self
    
    def clear_column(self, column: pd.Series, values_normal_values: list) -> pd.Series:
        # Функция для фильтрации значений в колонке по заданному множеству/диапозону

        new_column = np.array([])

        for value in column:
            if value in values_normal_values:
                new_column = np.append(new_column, value)
            
            else:
                new_column = np.append(new_column, pd.NA)

        return pd.Series(new_column)
    
    def clear_ejection(self):
        # Функция для удаления выбросов из данных

        self.data['Age'] = self.clear_column(self.data['Age'], list(range(1, 100)))
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

        self.data['Delay_from_due_date'] = self.clear_column(self.data['Delay_from_due_date'], list(range(0, max(self.data['Delay_from_due_date'].values))))
        self.data['Delay_from_due_date'] = self.data['Delay_from_due_date'].fillna(0)

        return self

    def drop_usless_columns(self):
        # Удаление признаков, не влияющих на таргет

        self.data = self.data.drop(['Annual_Income', 'Monthly_Balance', 'Occupation', 'Month', 'Total_EMI_per_month', 'Credit_Utilization_Ratio'], axis=1)

        return self

    def feature_engineering(self):
        # Генерация признаков, указывающих на динамику изменения признака относительно его значений на протяжении всех месяцов 

        self.data['Salary_Vector'] = self.data.groupby('SSN')['Monthly_Inhand_Salary'].rank()
        self.data['Credit_Card_Usage_Vector'] = self.data.groupby('SSN')['Credit_Utilization_Ratio'].rank()

        return self

    def decode_target(self):
        # Преобразование таргета с помощью хэш-таблицы

        target_map = {
            'Poor': 0,
            'Standard': 1,
            'Good': 2
        }

        self.data['Credit_Score'] = self.data['Credit_Score'].replace(target_map)

        return self
    
    def transform(self) -> pd.DataFrame:

        # См data_preprocessing.ipynb
        self.fill_null()
        self.to_type()
        self.cat_decode()
        self.clear_ejection()

        # См EDA.ipynb
        self.drop_usless_columns()

        # См feature_engineering.ipynb
        self.feature_engineering()

        # Удаляем признак, по которому мы идентефицировали клиента, так как он нам больше не нужен
        self.data = self.data.drop('SSN', axis=1)

        # Так же из data_preprocessing.ipynb
        if self.is_train:
            self.decode_target()
        
            X = self.data.drop('Credit_Score', axis=1)
            y = self.data['Credit_Score']

            return X, y
        
        else:
            return self.data
