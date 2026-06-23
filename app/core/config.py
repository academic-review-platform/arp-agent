from pydantic import BaseModel


class Settings(BaseModel):
    project_name: str = "ARP Agent API"
    description: str = (
        "API base para un agente de revision de documentos mediante RAG."
    )
    version: str = "0.1.0"


settings = Settings()
