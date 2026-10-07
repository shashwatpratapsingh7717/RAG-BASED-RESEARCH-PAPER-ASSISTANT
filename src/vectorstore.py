import json
from pathlib import Path
from typing import List
import faiss
import numpy as np

class FaissVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        dimension: int
    ):
        self.persist_dir = Path(
            persist_dir
        )
        self.persist_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        self.index_path = (
            self.persist_dir /
            "faiss.index"
        )
        self.metadata_path = (
            self.persist_dir /
            "metadata.json"
        )
        self.dimension = dimension
        self.index = None
        self.metadata = []

    # BUILD
    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[dict]
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D NumPy array."
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, got "
                f"{embeddings.shape[1]}"
            )
        if len(metadata) != len(embeddings):
            raise ValueError(
                "Metadata count must match "
                "embedding count."
            )
# Always create a NEW index
# This prevents duplicate vectors
# # when the user rebuilds the knowledge base.

        self.index = faiss.IndexFlatIP(
            self.dimension
        )
        self.index.add(
            embeddings.astype("float32")
        )
        self.metadata = list(metadata)
        self.save()
        print(
            f"[INFO] Built FAISS index with "
            f"{self.index.ntotal} vectors."
        )
# SAVE

    def save(self) -> None:
        if self.index is None:
            raise ValueError(
                "Cannot save an empty FAISS index."
            )

        faiss.write_index(
            self.index,
            str(self.index_path)
        )
        self.metadata_path.write_text(
            json.dumps(
                self.metadata,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )
        print(
            "[INFO] FAISS index saved."
        )

# LOAD
    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                "FAISS index not found."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Metadata file not found."
            )
        self.index = faiss.read_index(
            str(self.index_path)
        )
        self.metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8"
            )
        )
        print(
            f"[INFO] FAISS index loaded "
            f"with {self.index.ntotal} vectors."
        )

# SEARCH
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[dict]:
        if self.index is None:
            raise ValueError(
                "FAISS index is not loaded."
            )
        if self.index.ntotal == 0:
            return []
        top_k = min(
            top_k,
            self.index.ntotal
        )
        scores, indices = (
            self.index.search(
                query_embedding.astype("float32"),
                top_k
            )
        )

        results = []
        for score, index in zip(
            scores[0],
            indices[0]
        ):
            if index < 0:
                continue
            metadata = self.metadata[index]
            result = dict(metadata)
            result["score"] = float(
                score
            )

# Since embeddings are normalized,
# 1 - cosine similarity works as
# a simple distance-like value.
            result["distance"] = float(
                1.0 - score
            )
            results.append(result)
        return results
 # CLEAR KNOWLEDGE BASE
    def clear(self) -> None:
        """
        Completely clears the FAISS vector store.
        Removes:
        1. In-memory FAISS index
        2. In-memory metadata
        3. Persistent FAISS index file
        4. Persistent metadata file
        """
        print(
            "[INFO] Clearing FAISS knowledge base..."
        )
# Reset in-memory state
        self.index = None
        self.metadata = []

# Delete FAISS index
        if self.index_path.exists():
            self.index_path.unlink()
            print(
                "[INFO] Deleted FAISS index."
            )
# Delete metadata
        if self.metadata_path.exists():
            self.metadata_path.unlink()
            print(
                "[INFO] Deleted metadata."
            )
        print(
            "[INFO] FAISS knowledge base cleared."
        )
# PROPERTIES
    @property
    def is_ready(self) -> bool:
        return (
            self.index is not None
            and self.index.ntotal > 0
            and len(self.metadata) > 0
        )
    @property
    def size(self) -> int:
        if self.index is None:
            return 0
        return int(
            self.index.ntotal
        )