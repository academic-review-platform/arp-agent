import dspy

from app.core.config import settings
from app.services.vector_store import BibliographyVectorStore


class DocumentReviewSignature(dspy.Signature):
    """
    Evaluate an academic document using the provided reference sources.

    You have access to two tools:
    - fetch_bibliography_summary: retrieves the summary of a bibliography by ID.
      Use this first to decide if the bibliography is relevant.
    - fetch_bibliography_content: retrieves the full content of a bibliography by ID.
      Use this only if the summary indicates the source is relevant.

    Sessions are plain text references provided directly in the input.

    Steps:
    1. For each bibliography ID, fetch its summary and decide if it is relevant to the document.
    2. For relevant bibliographies, fetch the full content to use as a source.
    3. Use all relevant sources (bibliography content + sessions) to evaluate the document.
    4. Assign a score from 1 to 100 and write a comment in the same language as the document.
       The comment must explain what is good, what can be improved, and justify the score.
    """

    document: str = dspy.InputField(desc="The document to be evaluated.")
    sessions: str = dspy.InputField(
        desc="Reference session content, one per line, provided as plain text."
    )
    bibliography_ids: str = dspy.InputField(
        desc="Comma-separated bibliography IDs available in the database."
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

        def fetch_bibliography_summary(bibliography_id: str) -> str:
            """
            Fetch the summary of a bibliography from the database.
            Returns the summary text, or a message indicating it was not found.
            Use this to decide whether the full content is worth retrieving.
            """
            summary = self._vector_store.get_summary(bibliography_id.strip())
            if summary is None:
                return f"[Not found] No bibliography with ID '{bibliography_id}' exists in the database."
            return summary

        def fetch_bibliography_content(bibliography_id: str) -> str:
            """
            Fetch the full content of a bibliography from the database.
            Only call this after confirming via the summary that the source is relevant.
            Returns the full text, or a message indicating it was not found.
            """
            content = self._vector_store.get_content(bibliography_id.strip())
            if content is None:
                return f"[Not found] No content for bibliography ID '{bibliography_id}'."
            return content

        self.react = dspy.ReAct(
            DocumentReviewSignature,
            tools=[fetch_bibliography_summary, fetch_bibliography_content],
            max_iters=20,
        )

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
        ids_text = ", ".join(bibliography_ids) if bibliography_ids else "(none)"

        with dspy.context(lm=lm):
            result = self.react(
                document=document,
                sessions=sessions_text,
                bibliography_ids=ids_text,
            )

        raw_score = result.score
        if isinstance(raw_score, str):
            raw_score = int("".join(filter(str.isdigit, raw_score)) or "50")
        score = max(1, min(100, int(raw_score)))

        return score, result.comment.strip()
