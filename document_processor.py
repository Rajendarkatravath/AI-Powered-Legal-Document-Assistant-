from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st

def process_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() for page in reader.pages)
        if text.strip():  # Check if text was extracted
            doc = Document(page_content=text, metadata={"filename": uploaded_file.name})
            documents.append(doc)
        else:
            st.warning(f"No text extracted from {uploaded_file.name}")
    return documents

def create_embeddings(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    if not splits:
        raise ValueError("No document splits were created.")
    return embeddings, splits

def setup_faiss_index(documents):
    embeddings, splits = create_embeddings(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# The rest of your Streamlit code goes here
