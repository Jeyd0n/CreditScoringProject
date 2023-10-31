import requests

import numpy as np

from modules.data.get_data import get_data
from modules.data.transform_data import transform_data


def send_request(X: np.array, y: np.array) -> list:
    prediction = requests.post(
        "http://127.0.0.1:3000/get_predict",
        headers={"content-type": "application/json"},
        data=f'{X.tolist()}'
    ).text

    predictions = []
    for value in prediction:
        if value in {'0', '1', '2'}:
            predictions.append(int(value))

    for true, predicted in zip(y, predictions):
        print(
            f'Истинный класс: {true}. \n Предсказанный класс: {predicted}'
        )


def main():
    dataframe = get_data(is_train=True)
    X, y = transform_data(
        dataframe,
        is_train=True
    )

    send_request(
        X=X.iloc[124:154, :].to_numpy(),
        y=y[124:154].to_numpy()
    )


if __name__ == '__main__':
    main()
