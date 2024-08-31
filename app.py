import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from document_processor import process_documents
from retrieval import setup_faiss_index
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit application interface
st.title("AI-Powered Legal Document Assistant")
st.write("Upload legal documents (PDF) and chat with their content for insights and summaries.")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

# Check if Groq API key is provided
if api_key:
    # Initialize Groq-based language model
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Initialize session history
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # File uploader for legal documents
    uploaded_files = st.file_uploader("Choose legal documents (PDF files)", type="pdf", accept_multiple_files=True)

    # Process uploaded documents
    if uploaded_files:
        documents = process_documents(uploaded_files)
        # Create embeddings and setup FAISS index
        vectorstore = setup_faiss_index(documents)
        retriever = vectorstore.as_retriever()

        # Setup LangChain history-aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Contextualize the user's query based on chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Setup Question Answering and Summarization Chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a summary or answer based on the provided legal document context."),
            MessagesPlaceholder("chat_history"),
            ("human", "Based on the following context: {context}, please answer the query: {input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to manage session history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat Interface
        user_input = st.text_input("Your legal query:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input, "context": documents},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the Groq API Key")
