import time
from typing import List, Tuple
from src.embeddings import EmbeddingModel
from src.vectorstore import FaissVectorStore

class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vectorstore: FaissVectorStore
    ):
        self.embedding_model = (
            embedding_model
        )

        self.vectorstore = (
            vectorstore
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[dict], float]:

        start_time = time.perf_counter()
        query_embedding = (
            self.embedding_model
            .encode_query(query)
        )

        results = (
            self.vectorstore.search(
                query_embedding,
                top_k=top_k
            )
        )
        latency = (
            time.perf_counter()
            - start_time
        )
        return results, latency