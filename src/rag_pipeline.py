import time
from typing import List
from src.chunker import chunk_pages
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    UPLOAD_DIR,
    VECTOR_DB_DIR,
)

from src.embeddings import EmbeddingModel
from src.llm import GroqLLM
from src.pdf_processor import load_pdf_folder
from src.retriever import Retriever
from src.vectorstore import FaissVectorStore
class RAGPipeline:
    def __init__(self):

# 1. EMBEDDING MODEL
        print("[INFO] Initializing embedding model...")
        self.embedding_model = EmbeddingModel(
            EMBEDDING_MODEL
        )

# 2. FAISS VECTOR STORE
        self.vectorstore = FaissVectorStore(
            VECTOR_DB_DIR,
            self.embedding_model.dimension
        )
# Load existing FAISS index if available
        if (
            self.vectorstore.index_path.exists()
            and
            self.vectorstore.metadata_path.exists()
        ):
            print(
                "[INFO] Existing FAISS store found. "
                "Loading..."
            )
            self.vectorstore.load()
        else:
            print(
                "[INFO] No existing FAISS store found."
            )
# 3. RETRIEVER
        self.retriever = Retriever(
            self.embedding_model,
            self.vectorstore
        )

# 4. LLM
        self.llm = GroqLLM(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL
        )
        print(
            "[INFO] RAG Pipeline initialized successfully."
        )

    # BUILD KNOWLEDGE BASE
    def build_from_folder(self) -> dict:
        print(
            "[INFO] Loading PDF documents..."
        )
        pages = load_pdf_folder(
            UPLOAD_DIR
        )

        if not pages:
            raise ValueError(
                "No readable PDF pages found "
                "in data/uploads."
            )
        print(
            f"[INFO] Loaded {len(pages)} pages."
        )

# CHUNKING
        print(
            "[INFO] Creating text chunks..."
        )
        chunks = chunk_pages(
            pages,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        if not chunks:
            raise ValueError(
                "No chunks were generated."
            )
        print(
            f"[INFO] Created {len(chunks)} chunks."
        )
# EMBEDDINGS
        texts = [
            chunk["text"]
            for chunk in chunks
        ]
        print(
            f"[INFO] Generating embeddings for "
            f"{len(texts)} chunks..."
        )
        embeddings = (
            self.embedding_model
            .encode_documents(texts)
        )

# METADATA
        metadata = [
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ]
# BUILD FAISS INDEX
        print(
            "[INFO] Building FAISS index..."
        )
        self.vectorstore.build(
            embeddings,
            metadata
        )
        print(
            "[INFO] Knowledge base built successfully."
        )

        return {
            "documents": len(
                {
                    item["source"]
                    for item in chunks
                }
            ),

            "pages": len(pages),
            "chunks": len(chunks),
            "embedding_dim":
                self.embedding_model.dimension,
            "vectors":
                self.vectorstore.size
        }
    # ASK QUESTION
    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:
        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )
        if not self.vectorstore.is_ready:
            raise ValueError(
                "Knowledge base is empty. "
                "Build it first."
            )
        start_time = time.perf_counter()

# RETRIEVAL
        results, retrieval_latency = (
            self.retriever.retrieve(
                question,
                top_k=top_k
            )
        )
        if not results:
            return {
                "answer":
                    "I could not find relevant "
                    "information in the uploaded papers.",
                "sources": [],
                "retrieval_latency":
                    retrieval_latency,
                "latency_seconds":
                    time.perf_counter()
                    - start_time
            }

 # BUILD CONTEXT
        context_blocks = []
        for i, result in enumerate(
            results,
            start=1
        ):

            context_blocks.append(
                f"""
[Source {i}]
Document: {result["source"]}
Page: {result["page"]}
Content:
{result["text"]}
""".strip()
            )
        context = "\n\n".join(
            context_blocks
        )

# GENERATE ANSWER
        answer = self.llm.generate(
            question=question,
            context=context
        )
        total_latency = (
            time.perf_counter()
            - start_time
        )
# SOURCES
        sources = [
            {
                "source":
                    result["source"],
                "page":
                    result["page"],
                "text":
                    result["text"],
                "score":
                    result["score"],
                "distance":
                    result["distance"]
            }
            for result in results
        ]
        return {
            "answer":
                answer,
            "sources":
                sources,
            "retrieval_latency":
                retrieval_latency,
            "latency_seconds":
                total_latency
        }


# SUMMARIZE DOCUMENT
    def summarize_document(
        self,
        source_name: str
    ) -> str:
        if not source_name:
            raise ValueError(
                "Please select a document."
            )
        print(
            f"[INFO] Summarizing: {source_name}"
        )
# GET DOCUMENT CHUNKS
        chunks = [
            item
            for item
            in self.vectorstore.metadata
            if item["source"] == source_name
        ]

        if not chunks:
            raise ValueError(
                f"Document not found: {source_name}"
            )

# Sort by page and chunk
        chunks.sort(
            key=lambda x: (
                x["page"],
                x["chunk_id"]
            )
        )
# IMPORTANT:
# DO NOT SEND THE ENTIRE PAPER TO GROQ
# Select representative chunks from
# beginning, middle and end.
        max_chunks = 12
        if len(chunks) <= max_chunks:
            selected_chunks = chunks
        else:
            step = (
                len(chunks) /
                max_chunks
            )
            indices = [
                int(i * step)
                for i in range(max_chunks)
            ]
            selected_chunks = [
                chunks[i]
                for i in indices
            ]
# BUILD COMPACT CONTEXT
        context_parts = []
        max_context_chars = 11000
        current_length = 0
        for chunk in selected_chunks:
            chunk_text = (
                f"Page {chunk['page']}:\n"
                f"{chunk['text']}"
            )

            if (
                current_length
                + len(chunk_text)
                > max_context_chars
            ):
                break
            context_parts.append(
                chunk_text
            )
            current_length += len(
                chunk_text
            )
        context = "\n\n".join(
            context_parts
        )

        if not context:
            raise ValueError(
                "Could not create summary context."
            )
        print(
            f"[INFO] Sending approximately "
            f"{len(context)} characters to LLM."
        )

#  ONE LLM CALL
        summary = self.llm.generate(
            question=(
                "Summarize this research paper "
                "based on the supplied sections."
            ),
            context=context,
            instruction="""
Create a concise but informative research
paper summary.
Use exactly these sections:
# 1. Problem
What problem does the paper address?
# 2. Methodology
What methodology or approach is used?
# 3. Main Findings
What are the important findings?
# 4. Limitations
What limitations are mentioned?
# 5. Future Work
What future directions are mentioned?
Important rules:
- Use only the supplied context.
- Do not invent information.
- If something is not available, say
  "Not mentioned in the retrieved content."
- Keep the answer concise.
"""
        )
        return summary

# COMPARE TWO PAPERS
    def compare_documents(
        self,
        paper_a: str,
        paper_b: str,
        comparison_request: str
    ) -> dict:

        if paper_a == paper_b:
            raise ValueError(
                "Please select two different papers."
            )

        if not comparison_request.strip():
            comparison_request = (
                "Compare the methodology, "
                "results, limitations and "
                "future work of both papers."
            )

        print(
            f"[INFO] Comparing:\n"
            f"Paper A: {paper_a}\n"
            f"Paper B: {paper_b}"
        )

# GET CHUNKS FROM EACH PAPER
        chunks_a = [
            item
            for item
            in self.vectorstore.metadata
            if item["source"] == paper_a
        ]
        chunks_b = [
            item
            for item
            in self.vectorstore.metadata
            if item["source"] == paper_b
        ]
        if not chunks_a:
            raise ValueError(
                f"Paper A not found: {paper_a}"
            )
        if not chunks_b:
            raise ValueError(
                f"Paper B not found: {paper_b}"
            )

# SORT CHUNKS
        chunks_a.sort(
            key=lambda x: (
                x["page"],
                x["chunk_id"]
            )
        )
        chunks_b.sort(
            key=lambda x: (
                x["page"],
                x["chunk_id"]
            )
        )
# SELECT REPRESENTATIVE CHUNKS

        def select_chunks(
            chunks,
            max_chunks=5
        ):
            if len(chunks) <= max_chunks:
                return chunks
            step = (
                len(chunks) /
                max_chunks
            )
            indices = [
                int(i * step)
                for i in range(max_chunks)
            ]
            return [
                chunks[i]
                for i in indices
            ]
        selected_a = select_chunks(
            chunks_a,
            max_chunks=5
        )
        selected_b = select_chunks(
            chunks_b,
            max_chunks=5
        )
# BUILD PAPER A CONTEXT
        context_a = "\n\n".join(
            f"Page {item['page']}:\n"
            f"{item['text']}"
            for item in selected_a
        )

# BUILD PAPER B CONTEXT
        context_b = "\n\n".join(
            f"Page {item['page']}:\n"
            f"{item['text']}"
            for item in selected_b
        )
# COMBINED CONTEXT
        context = f"""
PAPER A
Document:
{paper_a}
{context_a}

PAPER B
Document:
{paper_b}
{context_b}
""".strip()
# LLM COMPARISON
        answer = self.llm.generate(
            question=comparison_request,
            context=context,
            instruction="""
Compare Paper A and Paper B.
Use this structure:
## 1. Problem
Explain the problem addressed by each paper.
## 2. Methodology
Compare their methodologies.
## 3. Dataset
Compare datasets if mentioned.
## 4. Results
Compare reported results if available.
## 5. Limitations
Compare their limitations.
## 6. Future Work
Compare future directions.
Clearly identify which information
belongs to Paper A and Paper B.
Important:
- Use only the supplied context.
- Do not invent information.
- If information is unavailable,
  say "Not mentioned in the retrieved content."
"""
        )
# SOURCE
        sources = []
        for item in selected_a:
            sources.append({
                "source":
                    item["source"],
                "page":
                    item["page"],
                "score":
                    None
            })
        for item in selected_b:
            sources.append({
                "source":
                    item["source"],
                "page":
                    item["page"],
                "score":
                    None
            })
        return {
            "answer":
                answer,
            "sources":
                sources
        }
# LIST DOCUMENTS
    def list_documents(
        self
    ) -> List[str]:
        if not self.vectorstore.metadata:
            return []
        return sorted(
            {
                item["source"]
                for item
                in self.vectorstore.metadata
            }
        )
# CLEAR KNOWLEDGE BASE
    def clear_knowledge_base(self) -> None:
        """
        Completely clears the knowledge base.
        Deletes:
        1. FAISS index
        2. Metadata
        3. Uploaded PDF files
        4. In-memory vector data
        """
        import shutil
        from pathlib import Path
        print(
            "[INFO] Clearing knowledge base..."
        )

 # Clear FAISS
        self.vectorstore.clear
# Clear uploaded PDFs 
        upload_path = Path(
            UPLOAD_DIR
        )
        if upload_path.exists():
            for item in upload_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        print(
            "[INFO] Knowledge base cleared successfully."
        )
# GET STATISTICS
    def get_stats(self) -> dict:
        ready = self.vectorstore.is_ready
        return {
            "documents": (
                len(self.list_documents())
                if ready
                else 0
            ),
            "chunks": (
                len(self.vectorstore.metadata)
                if ready
                else 0
            ),
            "embedding_dim": (
                self.embedding_model.dimension
                if ready
                else 0
            ),
            "ready": ready
        }

