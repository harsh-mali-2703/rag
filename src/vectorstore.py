import pickle
from pathlib import Path
from typing import List

import faiss

from src.embedding import EmbeddingPipeline
from src.models import TextChunk


class FaissVectorStore:
    def __init__(self, persist_dir: str | Path = "vector_store/saved_index"):
        self.persist_dir = Path(persist_dir)
        self.index = None
        self.chunks: List[TextChunk] = []
        self.embedding = EmbeddingPipeline()

    @property
    def index_path(self):
        return self.persist_dir / "faiss.index"

    @property
    def chunks_path(self):
        return self.persist_dir / "chunks.pkl"

    def build_from_documents(self, documents: List[TextChunk]):
        if not documents:
            raise ValueError("No readable PDF text was found.")

        # The PDF pages are chunked and converted into embeddings before saving in FAISS.
        self.chunks = self.embedding.chunk_documents(documents)
        texts = [chunk.page_content for chunk in self.chunks]
        embeddings = self.embedding.embed_texts(texts).astype("float32")

        # FAISS stores the vectors and later finds the closest chunks to the question.
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.save()

    def save(self):
        # Saving the index means we do not need to process the PDFs again after refresh.
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, "wb") as file:
            pickle.dump(self.chunks, file)

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.chunks_path, "rb") as file:
            self.chunks = pickle.load(file)

    def search(self, question: str, top_k: int = 4) -> List[TextChunk]:
        if self.index is None:
            raise ValueError("Vector store has not been loaded or built.")

        # The question is embedded in the same format as the PDF chunks.
        query_vector = self.embedding.embed_texts([question]).astype("float32")
        distances, indexes = self.index.search(query_vector, top_k)

        results = []
        for idx in indexes[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results
