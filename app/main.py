from fastapi import FastAPI

from app.core.config import settings


app = FastAPI(
    title=settings.project_name,
    description=settings.description,
    version=settings.version,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
