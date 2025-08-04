import logging
import numpy as np
from typing import List, Dict, Optional, Union
import pickle
import os
from pathlib import Path

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicEmbeddingModel:
    """
    Arabic text embedding model using paraphrase-multilingual-mpnet-base-v2.
    Supports encoding text chunks into vector representations.
    """
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-mpnet-base-v2",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name of the sentence transformer model
            cache_dir (str): Directory to cache the model
            device (str): Device to run the model on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or self._get_best_device()
        self.model = None
        self.embedding_dim = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers is required but not installed. Please install it with: pip install sentence-transformers")
        
        self._load_model()
        logger.info(f"Initialized ArabicEmbeddingModel with {model_name} on {self.device}")
    
    def _get_best_device(self) -> str:
        """
        Get the best available device for the model.
        
        Returns:
            str: Device name
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text into vector representation.
        
        Args:
            text (str): Input text to encode
            normalize (bool): Whether to normalize the embedding
            
        Returns:
            np.ndarray: Text embedding vector
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return np.zeros(self.embedding_dim)
    
    def encode_batch(self, texts: List[str], 
                    batch_size: int = 32,
                    normalize: bool = True,
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode a batch of texts into vector representations.
        
        Args:
            texts (List[str]): List of texts to encode
            batch_size (int): Batch size for encoding
            normalize (bool): Whether to normalize embeddings
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Array of text embeddings
        """
        if not texts:
            return np.empty((0, self.embedding_dim))
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            
            logger.info(f"Encoded {len(texts)} texts into embeddings of shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def encode_chunks(self, chunks: List[Dict[str, str]], 
                     text_field: str = 'text',
                     batch_size: int = 32,
                     normalize: bool = True) -> List[Dict[str, Union[str, np.ndarray, Dict]]]:
        """
        Encode text chunks and add embeddings to the chunk data.
        
        Args:
            chunks (List[Dict[str, str]]): List of text chunks
            text_field (str): Field name containing the text
            batch_size (int): Batch size for encoding
            normalize (bool): Whether to normalize embeddings
            
        Returns:
            List[Dict[str, Union[str, np.ndarray, Dict]]]: Chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.get(text_field, '') for chunk in chunks]
        
        # Encode texts
        embeddings = self.encode_batch(
            texts, 
            batch_size=batch_size, 
            normalize=normalize
        )
        
        # Add embeddings to chunks
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = chunk.copy()
            enriched_chunk['embedding'] = embeddings[i]
            enriched_chunk['embedding_model'] = self.model_name
            enriched_chunk['embedding_dim'] = self.embedding_dim
            enriched_chunks.append(enriched_chunk)
        
        logger.info(f"Added embeddings to {len(enriched_chunks)} chunks")
        return enriched_chunks
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Normalize embeddings if not already normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         chunk_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar chunks to a query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            chunk_embeddings (List[np.ndarray]): List of chunk embeddings
            top_k (int): Number of top similar chunks to return
            
        Returns:
            List[tuple]: List of (index, similarity_score) tuples
        """
        if not chunk_embeddings:
            return []
        
        similarities = []
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_embeddings(self, chunks: List[Dict], filepath: str):
        """
        Save chunks with embeddings to a file.
        
        Args:
            chunks (List[Dict]): Chunks with embeddings
            filepath (str): Path to save the file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(chunks, f)
            
            logger.info(f"Saved {len(chunks)} chunks with embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, filepath: str) -> List[Dict]:
        """
        Load chunks with embeddings from a file.
        
        Args:
            filepath (str): Path to the embeddings file
            
        Returns:
            List[Dict]: Chunks with embeddings
        """
        try:
            with open(filepath, 'rb') as f:
                chunks = pickle.load(f)
            
            logger.info(f"Loaded {len(chunks)} chunks with embeddings from {filepath}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the embedding model.
        
        Returns:
            Dict[str, Union[str, int]]: Model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown')
        }

class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings to avoid recomputation.
    """
    
    def __init__(self, cache_dir: str = "embeddings_cache"):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir (str): Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized embedding cache at {self.cache_dir}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """
        Generate a cache key for text and model combination.
        
        Args:
            text (str): Input text
            model_name (str): Model name
            
        Returns:
            str: Cache key
        """
        import hashlib
        content = f"{text}_{model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache if available.
        
        Args:
            text (str): Input text
            model_name (str): Model name
            
        Returns:
            Optional[np.ndarray]: Cached embedding or None
        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def save_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """
        Save embedding to cache.
        
        Args:
            text (str): Input text
            model_name (str): Model name
            embedding (np.ndarray): Embedding to cache
        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")

if __name__ == "__main__":
    # Example usage
    embedding_model = ArabicEmbeddingModel()
    
    # Test with Arabic text
    arabic_texts = [
        "الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات",
        "معالجة اللغات الطبيعية فرع مهم من الذكاء الاصطناعي",
        "التعلم العميق يعتمد على الشبكات العصبية الاصطناعية"
    ]
    
    # Encode texts
    embeddings = embedding_model.encode_batch(arabic_texts)
    print(f"Encoded {len(arabic_texts)} texts into embeddings of shape {embeddings.shape}")
    
    # Test similarity
    similarity = embedding_model.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two texts: {similarity:.4f}")
    
    # Model info
    info = embedding_model.get_model_info()
    print(f"Model info: {info}")