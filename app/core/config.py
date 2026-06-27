from pydantic import BaseModel


class Settings(BaseModel):
    project_name: str = "ARP Agent API"
    description: str = (
        "API base para un agente de revision de documentos mediante RAG."
    )
    version: str = "0.1.0"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "qwen/qwen2.5-vl-7b"
    vector_db_path: str = ".data/chroma"
    vector_collection_name: str = "bibliographies"


settings = Settings()
