from langchain_chroma import Chroma
from document_processor import create_embeddings

def setup_faiss_index(documents):
    embeddings, splits = create_embeddings(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def retrieve_documents(query, vectorstore):
    # Retrieve relevant documents using FAISS
    retriever = vectorstore.as_retriever()
    return retriever.get_relevant_documents(query)
