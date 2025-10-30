# ✨ Gemini-Powered RAG Chatbot: Grounded Q&A via LangChain and Local Embeddings

A high-performance Retrieval-Augmented Generation (RAG) system designed to answer user questions strictly based on a custom PDF document (Term Insurance Guide), leveraging LangChain, a local embedding model, and Gemini 2.5 Flash.

<img width="324" height="253" alt="image" src="https://github.com/user-attachments/assets/701c9cdf-1057-4174-84db-33e4cda6564d" />

## ⚙️ Architecture and Data Flow

<img width="245" height="302" alt="image" src="https://github.com/user-attachments/assets/40139dea-f889-43b8-93fe-3530c2237470" />


This project utilizes the LangChain Expression Language (LCEL) to orchestrate a sophisticated RAG chain, ensuring answers are strictly grounded in the source PDF.



**Key Components:**
* **Source:** Custom PDF Document (Term Insurance Guide)
* **Embedding:** Local Hugging Face model for privacy and speed.
* **Vector Store:** [Specify your Vector DB, e.g., ChromaDB, FAISS]
* **Orchestration:** LangChain Expression Language (LCEL).
* **Generation:** **Gemini 2.5 Flash** for fast and accurate grounding.

  ## ✅ Key Features

* **Grounding:** Answers are strictly limited to the content of the `Term Insurance Guide.pdf`.
* **Performance:** Uses the fast **Gemini 2.5 Flash** model.
* **Local Embeddings:** Ensures high-speed, private document processing without external API calls for embedding.
* **Extensible:** Built using LCEL for easy modification and scaling.

  ## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd gemini-rag-chatbot
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up API Key:**
    Create a file named `.env` in the root directory and add your key:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
4.  **Run the application:**
    ```bash
    python main.py
    ```

    ## 💡 Usage Example

    <img width="575" height="343" alt="image" src="https://github.com/user-attachments/assets/049342f4-ca29-4b4d-96ce-997f8651343a" />


**User Query:** "What are the eligibility criteria for the term insurance plan?"
**Model Response:** *[Show a clear, formatted example of the model's output]*

That's a perfect final step! Documenting these technical challenges is what transforms a successful coding session into a valuable portfolio piece that showcases your **problem-solving skills** to future employers.


## 🛠️ Technical Challenges and Solutions

This project was built to demonstrate a robust **Retrieval-Augmented Generation (RAG)** pipeline using a modern, modular stack. Execution required overcoming three significant environment and API configuration hurdles, which ultimately hardened the final solution.

### 1. Challenge: Environment and Dependency Lockout (Kaggle/Colab)

**Problem:** Initial development attempts were blocked by network restrictions (on Kaggle) and subsequent installation conflicts (`ModuleNotFoundError` for packages like `langchain-chroma` and `langchain-community`).

**Solution:**
* **Environment Migration:** The project was successfully migrated from a restrictive Kaggle environment to a stable **Google Colab** environment.
* **Modular Installation:** Implemented a targeted, robust installation strategy, explicitly installing the newly separated LangChain components (`langchain-core`, `langchain-community`) to ensure the modern dependency graph was resolved correctly. This eliminated persistent import errors.

---

### 2. Challenge: Gemini API Quota Exhaustion for Embeddings

**Problem:** During the crucial **Indexing Phase**, the project hit the strict free-tier limit for the `embed_content` metric, resulting in a persistent `ResourceExhausted: 429` error. This blocked the creation of the vector database.

**Solution:**
* **Decoupling:** The project was deliberately decoupled from the Gemini Embedding API to remove this single point of failure.
* **Local Model Integration:** The embedding process was switched from the remote API (`GoogleGenerativeAIEmbeddings`) to a **free, local, open-source model** (`HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`).
* **Result:** This solution ran the high-volume embedding task entirely on the Colab machine, bypassing the API quota limit and successfully completing the **Retrieval (R)** component of the RAG system.

---

### 3. Challenge: LLM API Alias and Configuration Errors

**Problem:** The final **Generation Phase** initially failed with a `404 Not Found` error, as the model alias `gemini-pro` was found to be retired or unsupported by the current version of the `langchain-google-genai` library.

**Solution:**
* **API Standardization:** The model name in the `ChatGoogleGenerativeAI` constructor was updated to the current, active alias: **`gemini-2.5-flash`**.
* **Performance Tuning:** The model was configured for reliable RAG performance by setting `temperature=0.0` (for deterministic, factual output) and increasing `max_output_tokens` (to ensure complete, untruncated answers).

---

### 🌟 Conclusion: Validation of Groundedness

The resulting pipeline was validated with a stringent **Anti-Hallucination Test** (asking "how many colors are in the rainbow?"). The chatbot successfully responded with: *"I cannot answer your question from the given source..."*
