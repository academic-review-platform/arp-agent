from pathlib import Path
import hashlib
import re
from typing import Any

import chromadb
import numpy as np

from app.core.config import settings


class LocalHashEmbeddingFunction:
    dimensions = 768

    @staticmethod
    def name() -> str:
        return "local_hash_embedding"

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        return [self._embed(document) for document in input]

    def _embed(self, document: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"\w+", document.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector


class BibliographyVectorStore:
    def __init__(self) -> None:
        db_path = Path(settings.vector_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.vector_collection_name,
            metadata={"description": "Bibliography content and generated summaries"},
            embedding_function=LocalHashEmbeddingFunction(),
        )

    def save(
        self,
        *,
        bibliography_id: str,
        content: str,
        summary: str,
    ) -> None:
        documents = [content, summary]
        ids = [
            f"{bibliography_id}:content",
            f"{bibliography_id}:summary",
        ]
        metadatas: list[dict[str, Any]] = [
            {
                "bibliography_id": bibliography_id,
                "document_type": "content",
            },
            {
                "bibliography_id": bibliography_id,
                "document_type": "summary",
            },
        ]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def get_summary(self, bibliography_id: str) -> str | None:
        """Retrieves the summary of a bibliography by its ID, or None if not found."""
        result = self.collection.get(ids=[f"{bibliography_id}:summary"])
        documents = result.get("documents") or []
        return documents[0] if documents else None

    def get_content(self, bibliography_id: str) -> str | None:
        """Retrieves the full content of a bibliography by its ID, or None if not found."""
        result = self.collection.get(ids=[f"{bibliography_id}:content"])
        documents = result.get("documents") or []
        return documents[0] if documents else None
