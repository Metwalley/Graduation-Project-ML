"""
Response formatter - adds disclaimers and citations
"""
from typing import List, Dict, Any


class ResponseFormatter:
    """Formats LLM responses"""
    
    @staticmethod
    def format_response(llm_response: str, source_articles: List[Dict[str, Any]]) -> str:
        """
        Format response (just return the answer, no sources)
        
        Args:
            llm_response: Raw response from LLM
            source_articles: Articles used (not shown to user)
            
        Returns:
            Clean response
        """
        # Just return the answer - no sources, no disclaimers
        return llm_response.strip()
