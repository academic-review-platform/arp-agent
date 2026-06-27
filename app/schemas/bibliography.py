from pydantic import BaseModel, ConfigDict, Field


class BibliographySummaryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bibliography_id: str = Field(..., alias="bibliographyId", min_length=1)
    content: str = Field(..., min_length=1)


class BibliographySummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bibliography_id: str = Field(..., alias="bibliographyId")
    summary: str

