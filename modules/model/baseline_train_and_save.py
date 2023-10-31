import sys

sys.path.append('.')


import bentoml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from modules.data.get_data import get_data
from modules.data.transform_data import transform_data


def model_train(X_train, y_train):
    """Инициализация пайплайна и его последующее обучение"""

    baseline = Pipeline([
        ('Scaler', StandardScaler()),
        ('Classifier', OneVsRestClassifier(RandomForestClassifier(random_state=42)))
    ])
    baseline.fit(
        X_train, y_train
    )

    return baseline


def save_model(baseline):
    """"
    Сохранение итоговой модели в локальный репозиторий BentoML
    и вывод тэг-а сохраненной модели в консоль
    """

    saved_model = bentoml.sklearn.save_model(
        name='baseline_classifier',
        model=baseline
    )

    print(f'Модель сохранена: {saved_model}')


def main():
    data = get_data(is_train=True)

    X, y = transform_data(
        dataset=data,
        is_train=True
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=1
    )

    model = model_train(X_train, y_train)
    save_model(model) #Тэг после первого запуска - Model(tag="baseline_classifier:jmg4qqtxqcphuqro")


if __name__ == '__main__':
    main()
