# Chatbot Service Documentation
**Project Component: Intelligent RAG-Based Medical Chatbot**

## 1. Abstract
This module implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist parents of children with developmental disorders (ADHD, Autism, Dyslexia). By leveraging a local Large Language Model (Llama 3.2 3B) via Ollama, the system provides context-aware, privacy-focused, and medically safe educational responses without relying on external cloud APIs.

## 2. System Architecture

The chatbot follows a modular RAG architecture:

### 2.1 Core Components
1.  **Knowledge Base (Vector Store):** 
    - Uses `FAISS` (Facebook AI Similarity Search) for efficient similarity search.
    - Documents are embedded using `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers), optimized for Arabic semantic understanding.
2.  **Retrieval Engine:**
    - Retrieves top-k relevant chunks based on cosine similarity with the user's query.
3.  **Generation Engine (LLM):**
    - Utilizing `Meta Llama 3.2 3B` (Quantized) running locally via `Ollama`.
    - Selected for its balance between performance (low latency on CPU/Consumer GPU) and reasoning capability in Arabic.
4.  **Orchestrator:**
    - Python-based pipeline managing the flow: Query -> Safety Filter -> Retrieval -> Prompt Construction -> Generation.

## 3. Implementation Details

### 3.1 Technology Stack
- **Language:** Python 3.9
- **LLM Server:** Ollama (Local Interference)
- **Containerization:** Docker & Docker Compose
- **Frameworks:** LangChain concepts (Custom Implementation), Sentence-Transformers, NumPy.
- **API Interface:** FastAPI (Internal), Streamlit (Demo UI).

### 3.2 Key Features
1.  **Privacy-First Design:** No data leaves the local environment (GDPR compliant architecture).
2.  **Cost-Efficiency:** Zero API costs; runs on local hardware.
3.  **Context Awareness:** Augments LLM responses with specific, verified medical articles (Seed Data).

## 4. Safety & Ethical AI

A critical component is the **Multi-Layer Safety Filter**, designed to prevent potential medical harm.

### 4.1 Filter Mechanism
- **Regex-Based Filtering:** Blocks explicit medical diagnosis requests (describe symptoms -> get diagnosis) and medication queries (dosage, prescriptions).
- **Educational Scope Enforcement:** The system prompt explicitly restricts the AI to an "Educational Assistant" role, refusing to act as a doctor.
- **Hallucination Control:** Prompt engineering enforces strict adherence to provided contexts ("Answer only from the provided sources").

**Example of Safety:**
- User: "What is the dose for Ritalin?" -> *Blocked: "I cannot provide medical prescriptions."*
- User: "How to manage ADHD behavior?" -> *Allowed: Educational strategies provided.*

## 5. Performance & Testing

### 5.1 Test Scenarios
The system was tested against 30+ real-world scenarios covering:
- **Domain Accuracy:** Detailed correct responses for ADHD, Autism, and Dyslexia.
- **Safety Compliance:** 100% block rate for medication requests.
- **Language Quality:** High-quality Arabic generation with no foreign language mixing (enforced via post-processing filters).

### 5.2 Latency
- **Average Response Time:** 3-5 seconds (on CPU), <1 second (on GPU).
- **Model Loading:** Instant (Model cached in Docker Volume).

## 6. Conclusion
The implementation successfully demonstrates a robust, offline-capable medical assistant that balances the power of Generative AI with the strict safety requirements of the healthcare domain.
