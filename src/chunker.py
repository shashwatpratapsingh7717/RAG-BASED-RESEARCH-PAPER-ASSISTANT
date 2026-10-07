from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(
    pages: List[dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n","\n",". "," ",""],)
    chunks = []

    for page in pages:
        text = page["text"]
        pieces = splitter.split_text(text)
        for chunk_id, piece in enumerate(pieces):
            cleaned_text = piece.strip()

            if not cleaned_text:
                continue

            chunks.append(
                {
                    "text": cleaned_text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": chunk_id,
                }
            )

    return chunks