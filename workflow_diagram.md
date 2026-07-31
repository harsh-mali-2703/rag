# Workflow Diagram

```text
User uploads PDFs
        |
        v
Read every PDF page with pypdf
        |
        v
Clean text and keep document/page metadata
        |
        v
Split text into chunks with overlap
        |
        v
Create embeddings using all-MiniLM-L6-v2
        |
        v
Store embeddings and chunks in FAISS
        |
        v
User asks a question
        |
        v
Retrieve top matching chunks
        |
        v
Send context and question to Groq LLM
        |
        v
Show answer with source document and page number
```
