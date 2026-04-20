"""
FastAPI Application - REST API for chatbot service
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from rag_engine import RAGEngine
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Medical Support Chatbot API",
    description="RAG-based educational chatbot for parents of children with developmental disorders",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine (global singleton)
rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    print("Starting chatbot service...")
    rag_engine = RAGEngine()
    print("Chatbot service ready!")


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request from client"""
    question: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class Source(BaseModel):
    """Article source information"""
    title: str
    category: str
    relevance: float


class ChatResponse(BaseModel):
    """Chat response to client"""
    answer: str
    sources: List[Source]
    blocked: bool
    reason: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Medical Support Chatbot",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "status": "healthy",
        "rag_engine": "ready",
        "model": settings.ollama_model
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Args:
        request: ChatRequest with question
        
    Returns:
        ChatResponse with answer and sources
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Process through RAG pipeline
        result = rag_engine.process_question(request.question)
        
        return ChatResponse(
            answer=result['answer'],
            sources=[Source(**source) for source in result['sources']],
            blocked=result['blocked'],
            reason=result['reason']
        )
    
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
