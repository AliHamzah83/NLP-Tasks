import logging
import uuid
from typing import List, Dict, Optional, Union, Any
import numpy as np
from pathlib import Path

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicVectorDB:
    """
    ChromaDB-based vector database for Arabic RAG system.
    Handles storage and retrieval of text embeddings with metadata.
    """
    
    def __init__(self, 
                 db_path: str = "chroma_db",
                 collection_name: str = "arabic_documents",
                 embedding_function: Optional[Any] = None):
        """
        Initialize the vector database.
        
        Args:
            db_path (str): Path to store the ChromaDB database
            collection_name (str): Name of the collection
            embedding_function: Custom embedding function (optional)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required but not installed. Please install it with: pip install chromadb")
        
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = embedding_function
        
        self._initialize_db()
        logger.info(f"Initialized ArabicVectorDB at {self.db_path} with collection '{self.collection_name}'")
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Arabic documents for RAG system"}
            )
            
            logger.info(f"Collection '{self.collection_name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, 
                     chunks: List[Dict[str, Any]], 
                     text_field: str = 'text',
                     embedding_field: str = 'embedding') -> bool:
        """
        Add document chunks to the vector database.
        
        Args:
            chunks (List[Dict[str, Any]]): List of document chunks with embeddings
            text_field (str): Field name containing the text
            embedding_field (str): Field name containing the embedding
            
        Returns:
            bool: Success status
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID
                chunk_id = chunk.get('id', str(uuid.uuid4()))
                ids.append(chunk_id)
                
                # Extract text
                text = chunk.get(text_field, '')
                documents.append(text)
                
                # Extract embedding
                embedding = chunk.get(embedding_field)
                if embedding is not None:
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    embeddings.append(embedding)
                else:
                    logger.warning(f"No embedding found for chunk {i}")
                    embeddings.append([0.0] * 768)  # Default embedding size
                
                # Extract metadata (exclude text and embedding)
                metadata = {}
                for key, value in chunk.items():
                    if key not in [text_field, embedding_field, 'id']:
                        # Convert numpy types to Python types for JSON serialization
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        elif isinstance(value, (np.int32, np.int64)):
                            value = int(value)
                        elif isinstance(value, (np.float32, np.float64)):
                            value = float(value)
                        elif isinstance(value, dict):
                            # Handle nested metadata
                            metadata.update(value)
                            continue
                        metadata[key] = value
                
                metadatas.append(metadata)
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(chunks)} documents to the vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector database: {e}")
            return False
    
    def search(self, 
               query_embedding: Union[np.ndarray, List[float]], 
               top_k: int = 5,
               where: Optional[Dict] = None,
               include: List[str] = None) -> Dict[str, List]:
        """
        Search for similar documents using embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k (int): Number of top results to return
            where (Dict): Metadata filter conditions
            include (List[str]): Fields to include in results
            
        Returns:
            Dict[str, List]: Search results with documents, distances, and metadata
        """
        try:
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Set default includes
            if include is None:
                include = ["documents", "distances", "metadatas"]
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=include
            )
            
            logger.info(f"Search completed, found {len(results.get('documents', [[]])[0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vector database: {e}")
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    
    def search_by_text(self, 
                      query_text: str,
                      embedding_model,
                      top_k: int = 5,
                      where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text query.
        
        Args:
            query_text (str): Query text
            embedding_model: Embedding model to encode query
            top_k (int): Number of top results to return
            where (Dict): Metadata filter conditions
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        try:
            # Encode query text
            query_embedding = embedding_model.encode_text(query_text)
            
            # Search using embedding
            results = self.search(query_embedding, top_k=top_k, where=where)
            
            # Format results
            formatted_results = []
            documents = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            for i, (doc, distance, metadata) in enumerate(zip(documents, distances, metadatas)):
                result = {
                    'rank': i + 1,
                    'text': doc,
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'distance': distance,
                    'metadata': metadata or {}
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search by text: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(limit=min(100, count))
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'db_path': str(self.db_path)
            }
            
            # Analyze metadata if available
            if sample_results.get('metadatas'):
                metadata_keys = set()
                for metadata in sample_results['metadatas']:
                    if metadata:
                        metadata_keys.update(metadata.keys())
                stats['metadata_fields'] = list(metadata_keys)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids (List[str]): List of document IDs to delete
            
        Returns:
            bool: Success status
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def update_documents(self, 
                        chunks: List[Dict[str, Any]], 
                        text_field: str = 'text',
                        embedding_field: str = 'embedding') -> bool:
        """
        Update existing documents in the database.
        
        Args:
            chunks (List[Dict[str, Any]]): List of document chunks to update
            text_field (str): Field name containing the text
            embedding_field (str): Field name containing the embedding
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare data for update
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for chunk in chunks:
                chunk_id = chunk.get('id')
                if not chunk_id:
                    logger.warning("Chunk missing ID for update, skipping")
                    continue
                
                ids.append(chunk_id)
                documents.append(chunk.get(text_field, ''))
                
                embedding = chunk.get(embedding_field)
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
                
                # Extract metadata
                metadata = {k: v for k, v in chunk.items() 
                           if k not in [text_field, embedding_field, 'id']}
                metadatas.append(metadata)
            
            # Update in ChromaDB
            self.collection.update(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Updated {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset (clear) the collection.
        
        Returns:
            bool: Success status
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Collection '{self.collection_name}' reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def export_data(self, output_path: str) -> bool:
        """
        Export all data from the collection.
        
        Args:
            output_path (str): Path to save the exported data
            
        Returns:
            bool: Success status
        """
        try:
            # Get all documents
            results = self.collection.get(
                include=["documents", "embeddings", "metadatas"]
            )
            
            import json
            
            # Prepare export data
            export_data = {
                'collection_name': self.collection_name,
                'total_documents': len(results.get('documents', [])),
                'documents': results.get('documents', []),
                'embeddings': results.get('embeddings', []),
                'metadatas': results.get('metadatas', [])
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported {export_data['total_documents']} documents to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    vector_db = ArabicVectorDB()
    
    # Sample data
    sample_chunks = [
        {
            'id': 'doc1_chunk1',
            'text': 'الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات',
            'embedding': np.random.rand(768).tolist(),
            'metadata': {'source': 'document1', 'chunk_id': 1}
        },
        {
            'id': 'doc1_chunk2', 
            'text': 'معالجة اللغات الطبيعية فرع مهم من الذكاء الاصطناعي',
            'embedding': np.random.rand(768).tolist(),
            'metadata': {'source': 'document1', 'chunk_id': 2}
        }
    ]
    
    # Add documents
    success = vector_db.add_documents(sample_chunks)
    print(f"Added documents: {success}")
    
    # Get stats
    stats = vector_db.get_collection_stats()
    print(f"Collection stats: {stats}")