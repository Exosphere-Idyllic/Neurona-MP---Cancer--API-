from app.domain.mp_neuron import MPNeuron
from app.schemas.prediction import PredictionRequest, PredictionResponse

class PredictionService:
    def __init__(self, model: MPNeuron):
        self._model = model

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        sample = {k.replace("_", " "): v for k, v in req.model_dump().items()}
        is_benign, score = self._model.predict(sample)
        return PredictionResponse(
            diagnosis="benigno" if is_benign else "maligno",
            score=score,
            threshold=self._model.threshold,
        )