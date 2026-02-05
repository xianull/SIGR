"""
True KG-RAG (Knowledge Graph Retrieval-Augmented Generation) Module

This module implements semantic retrieval-based KG-RAG:
1. TripletTextualizer: Convert KG triplets to natural language
2. IndexManager: Build and manage FAISS embedding index
3. RAGRetriever: Semantic retrieval of relevant triplets
4. RAGGenerator: Generate descriptions using retrieved context
"""

from .triplet_textualizer import TripletTextualizer
from .index_manager import IndexManager, TripletIndex, get_index_stats
from .retriever import RAGRetriever, RetrievedTriplet
from .rag_generator import RAGGenerator

__all__ = [
    'TripletTextualizer',
    'IndexManager',
    'TripletIndex',
    'get_index_stats',
    'RAGRetriever',
    'RetrievedTriplet',
    'RAGGenerator',
]
