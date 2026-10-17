from fastapi import APIRouter, Depends
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.api.deps import get_prediction_service

router = APIRouter(tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, service: PredictionService = Depends(get_prediction_service)):
    return service.predict(req)