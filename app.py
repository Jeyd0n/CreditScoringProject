import numpy as np
import bentoml
from bentoml.io import NumpyNdarray


baseline_classifier = bentoml.sklearn.get('baseline_classifier:latest').to_runner()

app = bentoml.Service(
    'baseline_service',
    runners=[baseline_classifier]
)


@app.api(input=NumpyNdarray(), output=NumpyNdarray())
def get_predict(input_batch: np.ndarray) -> np.ndarray:
    prediction = baseline_classifier.predict.run(input_batch)

    return prediction
