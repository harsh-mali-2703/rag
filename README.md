# Domain-Specific RAG Chatbot

The project i made is used for asking questions from uploaded PDF files. It uses pypdf for reading pages, a small chunking function, the all-MiniLM-L6-v2 embedding model, FAISS for vector search, Groq for answer generation, and Streamlit for the interface.

## Project Workflow

PDF upload -> text extraction -> chunking -> embeddings -> FAISS search -> LLM answer -> source display

## Folder Structure

```text
app.py
src/
  data_loader.py
  embedding.py
  models.py
  vectorstore.py
  search.py
  prompt.py
documents/
sample_documents/
vector_store/
requirements.txt
.env.example
workflow_diagram.md
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file and add:

```text
GROQ_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

Upload one or more PDFs from the sidebar, click **Process**, and then ask questions in the chat box. The answer also shows the document name and page number used as sources.

The `sample_documents` folder contains a sample PDF that can be used for checking the upload and question-answering flow.

## Notes

- Only PDF files are accepted.
- Empty pages are skipped.
- The app refuses to answer when the information is not found in the uploaded documents.
- Do not upload confidential documents without permission.
