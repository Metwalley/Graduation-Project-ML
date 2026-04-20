# System Architecture: Medical Diagnostic & Chatbot System

This diagram is designed to showcase your role as a **System Architect** to potential clients on Upwork. It highlights the entire data flow from the mobile app to the backend, and finally to your isolated, containerized AI engine.

## 🏛️ System Architecture Diagram

```mermaid
graph TD
    %% Define Styles
    classDef client fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef backend fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef ai_layer fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef database fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef model fill:#ffebee,stroke:#d32f2f,stroke-width:2px;

    %% Client Layer
    subgraph Client_Layer ["📱 Client Layer"]
        App["Mobile Application<br/>(Flutter)"]:::client
    end

    %% Backend Layer
    subgraph Backend_Layer ["⚙️ Core Backend Layer"]
        SpringAPI["Spring Boot REST API<br/>(Microservice)"]:::backend
        DB[("Relational DB<br/>(PostgreSQL/MySQL)")]:::database
    end

    %% AI Engine Layer (Your Work)
    subgraph AI_Engine_Layer ["🧠 AI Engine Layer (Dockerized Microservice)"]
        FastAPI["FastAPI Orchestrator<br/>(Python)"]:::ai_layer
        
        subgraph Diagnostic_Models ["Diagnostic Machine Learning"]
            AutismModel["Autism Prediction Model<br/>(99% Accuracy)"]:::model
            ADHDModel["ADHD Prediction Model<br/>(76% Accuracy)"]:::model
        end
        
        subgraph RAG_System ["RAG Chatbot Pipeline"]
            VectorDB[("FAISS Vector Database<br/>(Semantic Search)")]:::database
            Ollama["Local LLM Server<br/>(Ollama + Llama 3.2 3B)"]:::model
        end
    end

    %% Data Flow
    App -- "1. User Input / Medical Request" --> SpringAPI
    SpringAPI -- "Reads/Writes User Data" --> DB
    SpringAPI -- "2. Forwards Payload (REST POST)" --> FastAPI
    
    FastAPI -- "3a. Diagnostic Data" --> AutismModel
    FastAPI -- "3b. Diagnostic Data" --> ADHDModel
    
    FastAPI -- "4a. Natural Query" --> VectorDB
    VectorDB -- "4b. Context Items" --> FastAPI
    FastAPI -- "4c. Augmented Prompt" --> Ollama
    Ollama -- "4d. Generated Response" --> FastAPI
    
    AutismModel -- "Results" --> FastAPI
    ADHDModel -- "Results" --> FastAPI
    
    FastAPI -- "5. Unified JSON Response" --> SpringAPI
    SpringAPI -- "6. Formatted Output" --> App
```

### 💡 How to use this for Upwork:
1. **Render it:** You can copy the code block above and paste it into [Mermaid Live Editor](https://mermaid.live/), then download it as a high-resolution PNG or SVG!
2. **The Output:** It will generate a beautiful, color-coded diagram showing how the Frontend (Flutter), Backend (Spring Boot), and your AI Microservice (Docker+FastAPI+Ollama+RAG+ML Models) all communicate smoothly.
3. **The impact:** Clients will see you understand microservices, API contracts, local AI deployment, and enterprise architecture.
