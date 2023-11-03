import requests

import numpy as np

from modules.data.get_data import get_data
from modules.data.transform_data import transform_data
from requests_toolbelt.multipart.encoder import MultipartEncoder


def request_get_predict(X: np.array, y: np.array) -> list:
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


def request_show_metrics(X: np.array, y: np.array) -> tuple:
    input = MultipartEncoder(
        fields={
            # 'X': f'{X.tolist()}',
            # 'y': f'{y.tolist()}'
            'X': f'{X}',
            'y': f'{y}'
        }
    )

    metrics = requests.post(
        "http://127.0.0.1:3000/show_metrics",
        headers={"content-type": input.content_type},
        data=input
    ).text

    print(metrics)


def main():
    dataframe = get_data(is_train=True)
    X, y = transform_data(
        dataframe,
        is_train=True
    )

    request_get_predict(
        X=X.iloc[124:154, :].to_numpy(),
        y=y[124:154].to_numpy()
    )
    request_show_metrics(
        X=X.to_numpy().tolist(),
        y=y.to_numpy().tolist()
    )


if __name__ == '__main__':
    main()
