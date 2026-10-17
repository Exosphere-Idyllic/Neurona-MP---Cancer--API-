from typing import Literal
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    worst_perimeter: float = Field(gt=0, description="Peor perímetro observado")
    worst_radius: float = Field(gt=0, description="Peor radio observado")
    worst_concave_points: float = Field(ge=0, description="Peor número/proporción de puntos cóncavos")
    mean_concave_points: float = Field(ge=0, description="Media de puntos cóncavos")

class PredictionResponse(BaseModel):
    diagnosis: Literal["benigno", "maligno"]
    score: int
    threshold: int
    disclaimer: str = "Modelo educativo, no es una herramienta de diagnóstico."