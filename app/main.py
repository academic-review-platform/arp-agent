from fastapi import FastAPI

from app.api.bibliographies import router as bibliographies_router
from app.core.config import settings


app = FastAPI(
    title=settings.project_name,
    description=settings.description,
    version=settings.version,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(bibliographies_router)
