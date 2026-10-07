from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Embedding model loaded successfully.")

    def encode_documents(self,texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True,)
        return np.asarray(embeddings,dtype="float32")

    def encode_query(self,text: str) -> np.ndarray:
        embedding = self.model.encode([text],convert_to_numpy=True,normalize_embeddings=True,)
        return np.asarray(embedding,dtype="float32")
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()