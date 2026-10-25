"""Core bias detector: embeds text and measures group disparity."""
import numpy as np
from sentence_transformers import SentenceTransformer

class BiasDetector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load a sentence embedding model."""
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Convert a list of texts to normalized embedding vectors."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors

    def group_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """Average embedding representing a whole group."""
        return embeddings.mean(axis=0)

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors (already normalized = dot product)."""
        return float(np.dot(vec_a, vec_b))