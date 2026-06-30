import dspy

from app.core.config import settings
from app.services.vector_store import BibliographyVectorStore


class DocumentReviewSignature(dspy.Signature):
    """
    Evaluate an academic document using the provided reference sources.

    Use the bibliography summaries and session content to assess the document quality.
    Assign a score from 1 to 100 and write a detailed comment in the same language as
    the document explaining strengths, weaknesses, and improvement suggestions.
    """

    document: str = dspy.InputField(desc="The document to be evaluated.")
    sessions: str = dspy.InputField(
        desc="Reference session content, one per line, provided as plain text."
    )
    bibliographies: str = dspy.InputField(
        desc="Summaries of reference bibliographies, each separated by '---'."
    )
    score: int = dspy.OutputField(
        desc="Numeric score from 1 to 100 reflecting document quality based on sources."
    )
    comment: str = dspy.OutputField(
        desc=(
            "Detailed comment in the same language as the document. "
            "Covers strengths, weaknesses, and improvement suggestions."
        )
    )


class ReviewAgent(dspy.Module):
    def __init__(self, vector_store: BibliographyVectorStore) -> None:
        self._vector_store = vector_store
        self.predict = dspy.Predict(DocumentReviewSignature)

    def forward(
        self,
        document: str,
        sessions: list[str],
        bibliography_ids: list[str],
    ) -> tuple[int, str]:
        lm = dspy.LM(
            model=f"openai/{settings.lmstudio_model}",
            api_base=settings.lmstudio_base_url,
            api_key="lm-studio",
            cache=False,
        )

        sessions_text = "\n---\n".join(sessions) if sessions else "(none)"

        # Prefetch bibliography summaries before calling the model
        bib_parts: list[str] = []
        for bib_id in bibliography_ids:
            summary = self._vector_store.get_summary(bib_id.strip())
            if summary:
                bib_parts.append(f"[{bib_id}]\n{summary}")
        bibliographies_text = "\n---\n".join(bib_parts) if bib_parts else "(none)"

        with dspy.context(lm=lm):
            result = self.predict(
                document=document,
                sessions=sessions_text,
                bibliographies=bibliographies_text,
            )

        raw_score = result.score
        if isinstance(raw_score, str):
            raw_score = int("".join(filter(str.isdigit, raw_score)) or "50")
        score = max(1, min(100, int(raw_score)))

        return score, result.comment.strip()
