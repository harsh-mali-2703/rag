import os
import shutil
from pathlib import Path

import streamlit as st

from src.data_loader import load_pdf_documents
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore


DOCUMENTS_DIR = Path("documents")
VECTOR_DIR = Path("vector_store") / "saved_index"
MAX_FILE_SIZE_MB = 15


st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")


def reset_project_state():
    # This clears the old chat and removes the old vector index before processing again.
    st.session_state.messages = []
    st.session_state.rag = None
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)


def save_uploaded_files(uploaded_files):
    # Uploaded PDFs are saved locally so the loader can read them from one folder.
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        file_size_mb = uploaded_file.size / (1024 * 1024)

        # I only allow PDF files because this chatbot is made for PDF documents.
        if uploaded_file.type != "application/pdf" and not uploaded_file.name.lower().endswith(".pdf"):
            st.sidebar.warning(f"{uploaded_file.name} was skipped because it is not a PDF.")
            continue
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.sidebar.warning(f"{uploaded_file.name} is above {MAX_FILE_SIZE_MB} MB.")
            continue

        safe_name = Path(uploaded_file.name).name
        file_path = DOCUMENTS_DIR / safe_name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        saved_files.append(file_path)

    return saved_files


if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None


st.title("Domain-Specific RAG Chatbot")
st.caption("Upload PDFs, ask questions, and get answers with page sources.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Selected files:")
        for file in uploaded_files:
            st.write(f"- {file.name}")

    col1, col2 = st.columns(2)
    process_clicked = col1.button("Process", use_container_width=True)
    clear_clicked = col2.button("Clear", use_container_width=True)

    st.divider()
    top_k = st.slider("Chunks to retrieve", 3, 5, 4)
    st.caption("Please verify answers for important decisions.")

if clear_clicked:
    # Clear removes both uploaded documents and the saved FAISS index.
    reset_project_state()
    if DOCUMENTS_DIR.exists():
        shutil.rmtree(DOCUMENTS_DIR)
    st.rerun()

if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first.")
    else:
        reset_project_state()
        saved_files = save_uploaded_files(uploaded_files)
        if saved_files:
            with st.spinner("Reading PDFs and building the FAISS index..."):
                # First the PDFs are read, then embeddings are created and stored in FAISS.
                documents = load_pdf_documents(DOCUMENTS_DIR)
                store = FaissVectorStore(VECTOR_DIR)
                store.build_from_documents(documents)
                st.session_state.rag = RAGSearch(store)
            st.sidebar.success(f"Processed {len(saved_files)} file(s).")

if st.session_state.rag is None and VECTOR_DIR.exists():
    try:
        # This lets the app reuse the saved index if Streamlit refreshes the page.
        store = FaissVectorStore(VECTOR_DIR)
        store.load()
        st.session_state.rag = RAGSearch(store)
    except Exception:
        st.session_state.rag = None

# Showing old messages again makes 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"{source['source']} - page {source['page']}")

question = st.chat_input("Ask something from the uploaded PDFs")

if question:
    # The user's question is saved first so it stays visible after Streamlit reruns.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    if st.session_state.rag is None:
        answer = "Please upload and process at least one PDF before asking a question."
        sources = []
    elif not os.getenv("GROQ_API_KEY"):
        answer = "GROQ_API_KEY is missing. Add it to a .env file or your system environment."
        sources = []
    else:
        with st.spinner("Searching the documents..."):
            # RAGSearch finds useful chunks from FAISS and sends them to Groq for the final answer.
            answer, sources = st.session_state.rag.answer(question, top_k=top_k)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(f"{source['source']} - page {source['page']}")
