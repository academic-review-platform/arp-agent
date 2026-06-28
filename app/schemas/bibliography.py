from pydantic import BaseModel, ConfigDict, Field


class BibliographySummaryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bibliography_id: str = Field(..., alias="bibliographyId", min_length=1)
    content: str = Field(..., min_length=1)


class BibliographySummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bibliography_id: str = Field(..., alias="bibliographyId")
    summary: str


class DocumentReviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document: str = Field(..., min_length=1, description="Documento a ser evaluado.")
    sessions: list[str] = Field(
        default_factory=list,
        description="Contenido referencial de sesiones de clase.",
    )
    bibliography: list[str] = Field(
        default_factory=list,
        alias="bibliography",
        description="IDs de bibliografias presentes en la base de datos.",
    )


class DocumentReviewResponse(BaseModel):
    score: int = Field(..., ge=1, le=100)
    comment: str

