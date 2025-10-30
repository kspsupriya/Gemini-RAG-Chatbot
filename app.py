import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (Streamlit Cloud handles the .streamlit/secrets.toml equivalent)
# The actual key is passed via Streamlit Secrets, but we load environment variables for consistency
load_dotenv()

# --- Configuration ---
# IMPORTANT: This path must match your PDF's location in the GitHub repo
PDF_PATH = "data/source_document.pdf" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Get the API Key from environment variables (Streamlit Cloud injects it here)
# It must be named 'GEMINI_API_KEY' in Streamlit Secrets
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") 

if not GOOGLE_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in the Streamlit Cloud app secrets.")
    st.stop()


# --- CACHED FUNCTIONS (To run expensive parts only once) ---
# @st.cache_resource is critical for performance on Streamlit Cloud
@st.cache_resource
def setup_rag_system():
    # 1. Document Loading and Splitting
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error loading or splitting PDF. Ensure '{PDF_PATH}' is correct. Error: {e}")
        return None, None

    # 2. Embedding and Vector Store Setup
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create or load the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist() 

    # 3. RAG Chain Setup
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    You are an expert assistant for a "Term Insurance Guide". Your task is to answer questions
    ONLY based on the provided context. If the answer is not found in the context,
    state clearly that you cannot find the answer in the provided document.
    Do not invent information.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain, vectorstore


# --- Streamlit UI and Execution ---

def main():
    st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
    st.title("📄 Gemini RAG Chatbot: Grounded QA via LangChain")
    st.markdown("Ask a question about the `source_document.pdf` (Term Insurance Guide).")

    qa_chain, _ = setup_rag_system()

    if qa_chain is None:
        st.stop() # If RAG setup failed, stop the Streamlit app

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"]

                    sources = response.get("source_documents", [])
                    source_info = ""
                    if sources:
                        source_info = "\n\n**Sources Used:**\n"
                        for i, doc in enumerate(sources):
                            page = doc.metadata.get('page')
                            # Ensure page number is safe to display (in case metadata is missing)
                            page_display = f"Page {page + 1}" if page is not None else "Unknown Page"
                            source_info += f"- {page_display}: Content snippet: {doc.page_content[:150]}...\n"

                    full_response = answer + source_info

                    st.markdown(full_response)
                except Exception as e:
                    full_response = f"An error occurred while running the RAG chain: {e}"
                    st.error(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
