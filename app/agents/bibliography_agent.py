import dspy

from app.core.config import settings


class BibliographySummarySignature(dspy.Signature):
    """Resume el contenido de una bibliografia para uso academico."""

    bibliography_id: str = dspy.InputField(desc="Identificador de la bibliografia.")
    content: str = dspy.InputField(desc="Contenido completo de la bibliografia.")
    summary: str = dspy.OutputField(
        desc=(
            "Resumen en espanol, claro y estructurado, con los puntos centrales, "
            "tema, enfoque y utilidad para investigacion."
        )
    )


class BibliographyAgent(dspy.Module):
    def __init__(self) -> None:
        self.summarize = dspy.Predict(BibliographySummarySignature)

    def forward(self, bibliography_id: str, content: str) -> str:
        lm = dspy.LM(
            model=f"openai/{settings.lmstudio_model}",
            api_base=settings.lmstudio_base_url,
            api_key="lm-studio",
            cache=False,
        )
        with dspy.context(lm=lm):
            result = self.summarize(
                bibliography_id=bibliography_id,
                content=content,
            )

        return result.summary.strip()

