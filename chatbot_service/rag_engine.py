"""
RAG Engine - Orchestrates retrieval + generation
"""
from typing import Dict, Any
from knowledge_base import KnowledgeBase
from ollama_client import OllamaClient
from safety_filter import SafetyFilter
from response_formatter import ResponseFormatter
from config import settings


class RAGEngine:
    """Main RAG engine coordinating all components"""
    
    def __init__(self):
        """Initialize all RAG components"""
        print("Initializing RAG Engine...")
        
        # Initialize components
        self.safety_filter = SafetyFilter()
        
        self.knowledge_base = KnowledgeBase(
            articles_path=settings.knowledge_base_path,
            embedding_model_name=settings.embedding_model
        )
        
        self.ollama_client = OllamaClient(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature
        )
        
        self.response_formatter = ResponseFormatter()
        
        print("RAG Engine ready!")
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process user question through full RAG pipeline
        
        Args:
            question: User's question
            
        Returns:
            Response dict with answer, sources, metadata
        """
        # Step 1: Safety Check
        is_safe, reason = self.safety_filter.is_safe(question)
        if not is_safe:
            return {
                "answer": self.safety_filter.get_blocked_response(),
                "sources": [],
                "blocked": True,
                "reason": reason
            }
        
        # Step 2: Retrieve Relevant Articles
        relevant_articles = self.knowledge_base.search(
            query=question,
            top_k=settings.top_k_articles
        )
        
        # Step 3: Generate Response with Ollama
        llm_response = self.ollama_client.generate_response(
            question=question,
            context_articles=relevant_articles
        )
        
        # Step 4: Format with Disclaimers
        final_response = self.response_formatter.format_response(
            llm_response=llm_response,
            source_articles=relevant_articles
        )
        
        return {
            "answer": final_response,
            "sources": [
                {
                    "title": article['title'],
                    "category": article['category'],
                    "relevance": article['relevance_score']
                }
                for article in relevant_articles
            ],
            "blocked": False,
            "reason": "success"
        }
