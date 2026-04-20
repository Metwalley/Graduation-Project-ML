"""
Knowledge base loader and vector index builder
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path


class KnowledgeBase:
    """Loads articles and builds FAISS vector index for semantic search"""
    
    def __init__(self, articles_path: str, embedding_model_name: str):
        """
        Initialize knowledge base
        
        Args:
            articles_path: Path to articles.json
            embedding_model_name: HuggingFace model for embeddings
        """
        self.articles_path = Path(articles_path)
        self.articles: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        self.index: faiss.IndexFlatL2 = None
        
        # Load embedding model (supports Arabic)
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Load and index articles
        self._load_articles()
        self._build_index()
    
    def _load_articles(self):
        """Load articles from JSON file"""
        if not self.articles_path.exists():
            raise FileNotFoundError(f"Articles not found: {self.articles_path}")
        
        with open(self.articles_path, 'r', encoding='utf-8') as f:
            self.articles = json.load(f)
        
        print(f"Loaded {len(self.articles)} articles")
    
    def _build_index(self):
        """Build FAISS vector index from article embeddings"""
        if not self.articles:
            raise ValueError("No articles loaded")
        
        # Create searchable text for each article (title + summary + content)
        texts = [
            f"{article['title']} {article['summary']} {article['content'][:500]}"
            for article in self.articles
        ]
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index (L2 distance)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Built FAISS index with {self.index.ntotal} vectors (dim={dimension})")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant articles
        
        Args:
            query: User's question
            top_k: Number of articles to retrieve
            
        Returns:
            List of top-k most relevant articles with scores
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return articles with relevance scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            article = self.articles[idx].copy()
            article['relevance_score'] = float(1 / (1 + distance))  # Convert distance to similarity
            results.append(article)
        
        return results
