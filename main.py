import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma # Using ChromaDB as an example
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

# --- Configuration ---
# IMPORTANT: Ensure this matches the name and location of your uploaded PDF
PDF_PATH = "data/source_document.pdf" # Make sure this matches your uploaded file name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # A good local embedding model
CHROMA_DB_PATH = "./chroma_db" # Directory to store your vector database
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") # Retrieve API key

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in a .env file.")

# --- 1. Document Loading ---
def load_documents(pdf_path):
    """Loads a PDF document and splits it into chunks."""
    print(f"Loading document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# --- 2. Embedding and Vector Store ---
def create_vector_store(chunks, db_path):
    """
    Creates or loads a vector store from document chunks using local embeddings.
    """
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    # Using a local HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Check if a vector store already exists
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"Loading existing vector store from {db_path}...")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        print(f"Creating new vector store at {db_path}...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
        vectorstore.persist() # Save the vector store to disk
        print("Vector store created and saved.")
    return vectorstore

# --- 3. RAG Chain Setup ---
def setup_rag_chain(vectorstore):
    """
    Sets up the RetrievalQA chain with Gemini 2.5 Flash.
    """
    print("Setting up Gemini 2.5 Flash model...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)

    # Configure the retriever to fetch top 3 most relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Custom prompt template to guide the LLM
    # This prompt is crucial for anti-hallucination
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
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Combine retriever and LLM into a RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' means all retrieved docs are "stuffed" into the prompt
        retriever=retriever,
        return_source_documents=True, # Optional: to see which docs were used
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("RAG chain setup complete.")
    return qa_chain

# --- Main Execution Flow ---
def main():
    # 1. Load and process documents
    # IMPORTANT: Ensure PDF_PATH matches the name of your uploaded PDF
    chunks = load_documents(PDF_PATH)

    # 2. Create/Load Vector Store
    vectorstore = create_vector_store(chunks, CHROMA_DB_PATH)

    # 3. Setup RAG Chain
    qa_chain = setup_rag_chain(vectorstore)

    print("\n--- Chatbot Ready ---")
    print("Type 'exit' to quit.")

    # --- Interactive Q&A Loop ---
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        print("Searching for an answer...")
        response = qa_chain.invoke({"query": query})

        print("\n--- Answer ---")
        print(response["result"])

        # Optional: Print source documents to show where the answer came from
        # print("\n--- Source Documents Used ---")
        # for i, doc in enumerate(response["source_documents"]):
        #     print(f"Doc {i+1}: {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
        #     print(f"Content snippet: {doc.page_content[:200]}...\n")


if __name__ == "__main__":
    main()
