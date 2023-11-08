import sys

sys.path.append('..')


import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from modules.data.get_data import get_data
from modules.data.transform_data import transform_data
from sklearn.metrics import precision_score, recall_score, f1_score


baseline_classifier = bentoml.sklearn.get('baseline_classifier:jmg4qqtxqcphuqro').to_runner()

app = bentoml.Service(
    'baseline_service',
    runners=[baseline_classifier]
)


@app.api(input=NumpyNdarray(), output=NumpyNdarray())
def get_predict(input_batch: np.ndarray) -> np.ndarray:
    predictions = baseline_classifier.predict.run(input_batch)

    return predictions


@app.api(input=Multipart(X=NumpyNdarray(), y=NumpyNdarray()), output=NumpyNdarray())
def show_metrics(X: np.ndarray, y: np.ndarray):
    predictions = baseline_classifier.predict.run(X)

    precision = precision_score(predictions, y, average='weighted')
    recall = recall_score(predictions, y, average='weighted')
    f1 = f1_score(predictions, y, average='weighted')

    return np.array([precision, recall, f1])
