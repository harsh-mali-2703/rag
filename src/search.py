from typing import Dict, List, Tuple

import os
from dotenv import load_dotenv
import requests

from src.prompt import build_prompt
from src.vectorstore import FaissVectorStore


load_dotenv()


class RAGSearch:
    def __init__(
        self,
        vectorstore: FaissVectorStore,
        llm_model: str = "openai/gpt-oss-20b",
    ):
        self.vectorstore = vectorstore
        self.llm_model = llm_model

    def _ask_groq(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing.")

        # Groq is used here as the LLM that writes the final answer from the retrieved context.
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Groq API request failed: {response.status_code} {response.text}"
            ) from exc
        return response.json()["choices"][0]["message"]["content"]

    def answer(self, question: str, top_k: int = 4) -> Tuple[str, List[Dict[str, str]]]:
        # First we retrieve the most relevant chunks from the vector database.
        chunks = self.vectorstore.search(question, top_k=top_k)
        if not chunks:
            return "I could not find this information in the uploaded documents.", []

        context_parts = []
        sources = []
        seen_sources = set()

        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown document")
            page = chunk.metadata.get("page", "Unknown")

            # The context includes source and page so the answer can mention where it came from.
            context_parts.append(
                f"Source: {source}, page {page}\n{chunk.page_content}"
            )

            source_key = (source, page)
            if source_key not in seen_sources:
                sources.append({"source": source, "page": page})
                seen_sources.add(source_key)

        # The prompt combines the user's question with the retrieved PDF text.
        prompt = build_prompt(question, "\n\n".join(context_parts))
        return self._ask_groq(prompt), sources
