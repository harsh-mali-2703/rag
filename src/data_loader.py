from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.models import TextChunk


def load_pdf_documents(folder_path: str | Path) -> List[TextChunk]:
    """Read PDFs page by page and keep simple source metadata."""
    folder = Path(folder_path)
    documents: List[TextChunk] = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        # Each page is stored separately so the answer can show the correct page number.
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            # This removes extra spaces and new lines that usually come from PDF text.
            text = " ".join(text.split())
            if not text:
                continue

            documents.append(
                TextChunk(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    return documents


if __name__ == "__main__":
    docs = load_pdf_documents("documents")
    print(f"Loaded {len(docs)} pages")
