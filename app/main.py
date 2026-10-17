from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.infrastructure.model_repository import load_model
from app.api.routes import health, predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(settings.model_path)
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="API Cáncer de mama (MP)", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/api/v1")
    return app

app = create_app()