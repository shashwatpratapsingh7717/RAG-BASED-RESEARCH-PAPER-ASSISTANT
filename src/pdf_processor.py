from pathlib import Path
from typing import List
import fitz  # PyMuPDF

def extract_pdf_pages(pdf_path: Path) -> List[dict]:
    """
    Extract text page-by-page from a PDF.
    Each returned item contains:
    - text
    - source
    - page
    """
    pages = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                {
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                }
            )

    return pages

def save_uploaded_pdfs(uploaded_files, upload_dir: Path) -> List[Path]:
    """
    Save Streamlit uploaded PDFs to data/uploads.
    """
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    saved_files = []
    for uploaded_file in uploaded_files:

        # Prevent unsafe path components.
        safe_name = Path(uploaded_file.name).name
        file_path = upload_dir / safe_name
        file_path.write_bytes(
            uploaded_file.getvalue()
        )

        saved_files.append(file_path)
    return saved_files


def load_pdf_folder(upload_dir: Path) -> List[dict]:
    """
    Load all PDFs from the upload directory.
    """
    all_pages = []
    pdf_files = sorted(
        upload_dir.glob("*.pdf")
    )

    for pdf_file in pdf_files:
        pages = extract_pdf_pages(pdf_file)

        all_pages.extend(pages)
    return all_pages