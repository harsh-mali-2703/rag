from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.models import TextChunk


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 900,
        chunk_overlap: int = 120,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
        self.model = AutoModel.from_pretrained(f"sentence-transformers/{model_name}")
        self.model.eval()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_documents(self, documents: List[TextChunk]) -> List[TextChunk]:
        chunks = []

        for document in documents:
            # LangChain's splitter breaks long pages into smaller overlapping chunks.
            for chunk_text in self.text_splitter.split_text(document.page_content):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunks.append(
                    TextChunk(
                        page_content=chunk_text,
                        metadata=document.metadata.copy(),
                    )
                )

        return chunks

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # The tokenizer converts normal text into numbers that the transformer model understands.
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = self.model(**encoded)

        # Mean pooling makes one embedding vector for each input text.
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
