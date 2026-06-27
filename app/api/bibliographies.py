from fastapi import APIRouter, HTTPException, status
from httpx import HTTPError

from app.agents.bibliography_agent import BibliographyAgent
from app.schemas.bibliography import (
    BibliographySummaryRequest,
    BibliographySummaryResponse,
)
from app.services.vector_store import BibliographyVectorStore


router = APIRouter(prefix="/bibliographies", tags=["bibliographies"])


@router.post(
    "/summarize",
    response_model=BibliographySummaryResponse,
    status_code=status.HTTP_200_OK,
)
def summarize_bibliography(
    request: BibliographySummaryRequest,
) -> BibliographySummaryResponse:
    print(f"Received request to summarize bibliography {request.bibliography_id}")
    agent = BibliographyAgent()
    vector_store = BibliographyVectorStore()

    try:
        summary = agent(
            bibliography_id=request.bibliography_id,
            content=request.content,
        )
        vector_store.save(
            bibliography_id=request.bibliography_id,
            content=request.content,
            summary=summary,
        )
    except HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach LM Studio: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return BibliographySummaryResponse(
        bibliographyId=request.bibliography_id,
        summary=summary,
    )

