"""
Arabic RAG (Retrieval-Augmented Generation) System

A comprehensive Arabic language RAG system with the following components:
- Data collection with configurable paths
- Arabic text preprocessing (noise removal, normalization, tokenization)
- Chunking strategies (fixed-size and sentence-based)
- Multilingual embedding model (paraphrase-multilingual-mpnet-base-v2)
- ChromaDB vector database integration
- AraGPT2 for Arabic text generation
- Complete RAG pipeline with evaluation metrics
"""

__version__ = "1.0.0"
__author__ = "Arabic RAG System"
__description__ = "Complete Arabic RAG system for document retrieval and generation"

# Import main classes
from .data_collection import ArabicDataCollector
from .preprocessing import ArabicTextPreprocessor, PreprocessingConfig
from .chunking import ArabicTextChunker, ChunkingConfig
from .embeddings import ArabicEmbeddingModel, EmbeddingCache
from .vector_db import ArabicVectorDB
from .aragpt2_model import AraGPT2Generator, AraGPT2Pipeline
from .rag_pipeline import ArabicRAGPipeline, RAGEvaluator

# Import evaluation module
from .evaluation import ArabicRAGEvaluator, EvaluationMetrics

__all__ = [
    # Data handling
    'ArabicDataCollector',
    
    # Text processing
    'ArabicTextPreprocessor',
    'PreprocessingConfig',
    
    # Chunking
    'ArabicTextChunker', 
    'ChunkingConfig',
    
    # Embeddings
    'ArabicEmbeddingModel',
    'EmbeddingCache',
    
    # Vector database
    'ArabicVectorDB',
    
    # Language model
    'AraGPT2Generator',
    'AraGPT2Pipeline',
    
    # Main pipeline
    'ArabicRAGPipeline',
    'RAGEvaluator',
    
    # Evaluation
    'ArabicRAGEvaluator',
    'EvaluationMetrics'
]