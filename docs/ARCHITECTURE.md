# System Architecture Overview

## Question: Are These Two Separate APIs?

**YES! You have TWO completely separate microservices:**

---

## 🏥 Service 1: Diagnosis API (Already Built - FastAPI in Docker)

**What it does:** 
- Predicts ADHD, Autism, Dyslexia from questionnaire answers
- Uses ML models (XGBoost, Random Forest)
- Makes diagnoses

**Technology:**
- FastAPI (Python)
- Docker container
- ML models (trained on diagnostic datasets)

**Endpoints:**
```
POST /predict
- Accepts: questionnaire answers + age + sex
- Returns: diagnosis prediction + confidence
```

**Status:** ✅ Already sent to team (Docker image)

---

## 💬 Service 2: Chatbot API (Just Built - NEW)

**What it does:**
- Answers parent questions about disorders
- Educational support only (NOT diagnosis)
- RAG-based (retrieves from articles, generates responses)

**Technology:**
- FastAPI (Python)
- Gemini API
- FAISS vector database
- Sentence transformers

**Endpoints:**
```
POST /chat
- Accepts: parent question (text)
- Returns: educational answer + sources + disclaimers

GET /health
- Returns: service health status
```

**Status:** ✅ Just completed (this new service)

---

## 🏗️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUTTER MOBILE APP                        │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │ Diagnostic │  │  Results   │  │  Parent Chat         │  │
│  │ Screens    │  │  Screen    │  │  (Ask Questions)     │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           │               │                      │
           │               │                      │
           ▼               ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│              SPRING BOOT BACKEND (Gateway)                   │
│  ┌────────────────────┐         ┌──────────────────────┐   │
│  │ /api/diagnose      │         │  /api/chatbot/ask    │   │
│  │ (proxy to AI)      │         │  (proxy to chatbot)  │   │
│  └────────────────────┘         └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           │                                    │
           │                                    │
           ▼                                    ▼
┌────────────────────────┐          ┌────────────────────────┐
│  SERVICE 1 (EXISTING)  │          │  SERVICE 2 (NEW)       │
│  Diagnosis API         │          │  Chatbot API           │
│  ────────────────────  │          │  ──────────────────    │
│  FastAPI (Port 8000)   │          │  FastAPI (Port 8001)   │
│  Docker Container      │          │  Docker Container      │
│                        │          │                        │
│  POST /predict         │          │  POST /chat            │
│  - ML Models           │          │  - RAG Engine          │
│  - ADHD (XGBoost)      │          │  - Gemini API          │
│  - Autism (XGBoost)    │          │  - FAISS Search        │
│  - Dyslexia (RF)       │          │  - Safety Filters      │
│                        │          │                        │
│  Returns: Diagnosis    │          │  Returns: Answer       │
└────────────────────────┘          └────────────────────────┘
```

---

## 🔄 How They Work Together

### User Flow 1: Diagnosis (Existing)
```
1. Parent fills diagnostic questionnaire in Flutter
2. Flutter → Spring Boot → Diagnosis API (Port 8000)
3. ML model predicts disorder
4. Result shown to parent
```

### User Flow 2: Ask Questions (NEW)
```
1. Parent types question in chat screen (Flutter)
2. Flutter → Spring Boot → Chatbot API (Port 8001)
3. RAG retrieves articles + Gemini generates answer
4. Answer shown to parent with disclaimers
```

---

## 📦 Deployment

Both services run **independently**:

```yaml
# docker-compose.yml (Your team will have)
services:
  diagnosis-api:
    image: diagnosis-fastapi:latest
    ports:
      - "8000:8000"
  
  chatbot-api:
    image: chatbot-fastapi:latest
    ports:
      - "8001:8001"
```

---

## ✅ Summary

| Aspect | Diagnosis API | Chatbot API |
|--------|--------------|-------------|
| **Purpose** | Diagnose disorders | Answer questions |
| **Input** | Questionnaire | Text question |
| **Output** | Prediction | Educational answer |
| **Port** | 8000 | 8001 |
| **Status** | ✅ Sent to team | ✅ Just completed |
| **Docker** | Already exists | Just created |

**They are completely separate services that your Spring Boot backend will call based on what the user is doing in the app.**

---

## 🚀 What You Need to Send Your Team

1. **Chatbot Docker image** (or docker-compose.yml)
2. **Integration instructions** (in README.md)
3. **API documentation** (Swagger at /docs)

The Spring Boot team just needs to add **one new controller** to proxy chatbot requests, similar to how they already proxy diagnosis requests.
