import logging
import time
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

# Import our custom modules
from .data_collection import ArabicDataCollector
from .preprocessing import ArabicTextPreprocessor, PreprocessingConfig
from .chunking import ArabicTextChunker, ChunkingConfig
from .embeddings import ArabicEmbeddingModel
from .vector_db import ArabicVectorDB
from .aragpt2_model import AraGPT2Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicRAGPipeline:
    """
    Complete Arabic RAG (Retrieval-Augmented Generation) pipeline.
    Integrates all components for end-to-end question answering.
    """
    
    def __init__(self,
                 data_dir: str = "data",
                 db_path: str = "chroma_db",
                 collection_name: str = "arabic_documents",
                 embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
                 llm_model_name: str = "aubmindlab/aragpt2-base",
                 chunking_strategy: str = "sentence_based",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize the Arabic RAG pipeline.
        
        Args:
            data_dir (str): Directory containing data files
            db_path (str): Path for ChromaDB storage
            collection_name (str): Name of the vector database collection
            embedding_model_name (str): Name of the embedding model
            llm_model_name (str): Name of the LLM model
            chunking_strategy (str): Chunking strategy ("fixed_size" or "sentence_based")
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.data_dir = data_dir
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize components
        logger.info("Initializing Arabic RAG Pipeline components...")
        
        # Data collection
        self.data_collector = ArabicDataCollector(data_dir)
        
        # Text preprocessing
        self.preprocessing_config = PreprocessingConfig()
        self.preprocessor = ArabicTextPreprocessor(self.preprocessing_config)
        
        # Text chunking
        self.chunking_config = ChunkingConfig(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunker = ArabicTextChunker(self.chunking_config)
        
        # Embedding model
        self.embedding_model = ArabicEmbeddingModel(model_name=embedding_model_name)
        
        # Vector database
        self.vector_db = ArabicVectorDB(
            db_path=db_path,
            collection_name=collection_name
        )
        
        # Language model
        self.llm_pipeline = AraGPT2Pipeline(model_name=llm_model_name)
        
        logger.info("Arabic RAG Pipeline initialized successfully")
    
    def build_knowledge_base(self, 
                           force_rebuild: bool = False,
                           text_column: str = "text") -> Dict[str, Any]:
        """
        Build the knowledge base from data files.
        
        Args:
            force_rebuild (bool): Whether to rebuild even if data exists
            text_column (str): Column name for text in CSV files
            
        Returns:
            Dict[str, Any]: Build statistics
        """
        start_time = time.time()
        logger.info("Starting knowledge base construction...")
        
        # Check if knowledge base already exists
        stats = self.vector_db.get_collection_stats()
        if stats.get('total_documents', 0) > 0 and not force_rebuild:
            logger.info(f"Knowledge base already exists with {stats['total_documents']} documents")
            return stats
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data files...")
            documents = self.data_collector.load_all_data(text_column=text_column)
            
            if not documents:
                # Create sample data if no documents found
                logger.info("No documents found, creating sample data...")
                self.data_collector.create_sample_data()
                documents = self.data_collector.load_all_data(text_column=text_column)
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Step 2: Preprocess documents
            logger.info("Step 2: Preprocessing documents...")
            preprocessed_docs = self.preprocessor.preprocess_documents(documents)
            
            # Step 3: Chunk documents
            logger.info("Step 3: Chunking documents...")
            chunks = self.chunker.chunk_documents(preprocessed_docs)
            
            if not chunks:
                raise ValueError("No chunks created from documents")
            
            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings...")
            enriched_chunks = self.embedding_model.encode_chunks(chunks)
            
            # Step 5: Store in vector database
            logger.info("Step 5: Storing in vector database...")
            success = self.vector_db.add_documents(enriched_chunks)
            
            if not success:
                raise ValueError("Failed to store documents in vector database")
            
            # Get final statistics
            final_stats = self.vector_db.get_collection_stats()
            chunk_stats = self.chunker.get_chunk_statistics(chunks)
            
            build_time = time.time() - start_time
            
            result = {
                'success': True,
                'build_time_seconds': round(build_time, 2),
                'total_documents': len(documents),
                'total_chunks': len(chunks),
                'vector_db_stats': final_stats,
                'chunk_stats': chunk_stats,
                'embedding_model': self.embedding_model.model_name,
                'chunking_strategy': self.chunking_config.strategy
            }
            
            logger.info(f"Knowledge base built successfully in {build_time:.2f} seconds")
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to build knowledge base: {e}")
            return {
                'success': False,
                'error': str(e),
                'build_time_seconds': time.time() - start_time
            }
    
    def query(self, 
              question: str,
              top_k: int = 5,
              max_contexts: int = 3,
              similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question (str): Question to answer
            top_k (int): Number of documents to retrieve
            max_contexts (int): Maximum contexts to use for generation
            similarity_threshold (float): Minimum similarity threshold
            
        Returns:
            Dict[str, Any]: Answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess query
            processed_question = self.preprocessor.preprocess_text(question)
            
            # Step 2: Retrieve relevant documents
            logger.info(f"Searching for: {question}")
            retrieved_docs = self.vector_db.search_by_text(
                query_text=processed_question,
                embedding_model=self.embedding_model,
                top_k=top_k
            )
            
            # Filter by similarity threshold
            filtered_docs = [
                doc for doc in retrieved_docs 
                if doc.get('similarity_score', 0) >= similarity_threshold
            ]
            
            if not filtered_docs:
                return {
                    'question': question,
                    'answer': 'لم أجد معلومات كافية للإجابة على هذا السؤال.',
                    'confidence': 0.0,
                    'retrieved_docs': [],
                    'processing_time': time.time() - start_time,
                    'success': True
                }
            
            # Step 3: Generate answer using LLM
            logger.info(f"Generating answer using {len(filtered_docs)} retrieved documents")
            result = self.llm_pipeline.answer_question(
                question=question,
                retrieved_docs=filtered_docs,
                max_contexts=max_contexts
            )
            
            # Add metadata
            result.update({
                'processing_time': time.time() - start_time,
                'retrieved_docs': filtered_docs,
                'success': True,
                'query_preprocessing': {
                    'original_question': question,
                    'processed_question': processed_question
                }
            })
            
            logger.info(f"Query processed in {result['processing_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'question': question,
                'answer': 'حدث خطأ في معالجة السؤال.',
                'error': str(e),
                'success': False,
                'processing_time': time.time() - start_time
            }
    
    def batch_query(self, 
                   questions: List[str],
                   **query_kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions (List[str]): List of questions
            **query_kwargs: Additional arguments for query method
            
        Returns:
            List[Dict[str, Any]]: List of results
        """
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            result = self.query(question, **query_kwargs)
            results.append(result)
        
        return results
    
    def add_documents(self, 
                     documents: List[Dict[str, str]],
                     text_field: str = 'text') -> bool:
        """
        Add new documents to the knowledge base.
        
        Args:
            documents (List[Dict[str, str]]): List of documents to add
            text_field (str): Field containing the text
            
        Returns:
            bool: Success status
        """
        try:
            # Preprocess documents
            preprocessed_docs = self.preprocessor.preprocess_documents(documents)
            
            # Chunk documents
            chunks = self.chunker.chunk_documents(preprocessed_docs)
            
            # Generate embeddings
            enriched_chunks = self.embedding_model.encode_chunks(chunks, text_field=text_field)
            
            # Add to vector database
            success = self.vector_db.add_documents(enriched_chunks)
            
            if success:
                logger.info(f"Added {len(documents)} documents ({len(chunks)} chunks) to knowledge base")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            vector_stats = self.vector_db.get_collection_stats()
            embedding_info = self.embedding_model.get_model_info()
            llm_info = self.llm_pipeline.generator.get_model_info()
            
            return {
                'vector_database': vector_stats,
                'embedding_model': embedding_info,
                'language_model': llm_info,
                'chunking_config': {
                    'strategy': self.chunking_config.strategy,
                    'chunk_size': self.chunking_config.chunk_size,
                    'chunk_overlap': self.chunking_config.chunk_overlap
                },
                'preprocessing_config': {
                    'remove_diacritics': self.preprocessing_config.remove_diacritics,
                    'normalize_alef': self.preprocessing_config.normalize_alef,
                    'remove_stopwords': self.preprocessing_config.remove_stopwords
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {'error': str(e)}
    
    def export_knowledge_base(self, output_path: str) -> bool:
        """
        Export the knowledge base to a file.
        
        Args:
            output_path (str): Path to save the export
            
        Returns:
            bool: Success status
        """
        try:
            return self.vector_db.export_data(output_path)
        except Exception as e:
            logger.error(f"Failed to export knowledge base: {e}")
            return False
    
    def reset_knowledge_base(self) -> bool:
        """
        Reset (clear) the knowledge base.
        
        Returns:
            bool: Success status
        """
        try:
            success = self.vector_db.reset_collection()
            if success:
                logger.info("Knowledge base reset successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to reset knowledge base: {e}")
            return False

class RAGEvaluator:
    """
    Evaluator for RAG system performance.
    """
    
    def __init__(self, rag_pipeline: ArabicRAGPipeline):
        """
        Initialize evaluator.
        
        Args:
            rag_pipeline (ArabicRAGPipeline): RAG pipeline to evaluate
        """
        self.rag_pipeline = rag_pipeline
    
    def evaluate_retrieval(self, 
                          test_queries: List[Dict[str, Any]],
                          top_k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            test_queries (List[Dict[str, Any]]): Test queries with expected results
            top_k (int): Number of documents to retrieve
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not test_queries:
            return {}
        
        total_precision = 0
        total_recall = 0
        total_queries = len(test_queries)
        
        for query_data in test_queries:
            question = query_data.get('question', '')
            expected_docs = set(query_data.get('relevant_docs', []))
            
            # Retrieve documents
            retrieved_docs = self.rag_pipeline.vector_db.search_by_text(
                query_text=question,
                embedding_model=self.rag_pipeline.embedding_model,
                top_k=top_k
            )
            
            retrieved_ids = set([
                doc.get('metadata', {}).get('document_id', '') 
                for doc in retrieved_docs
            ])
            
            # Calculate precision and recall
            if retrieved_ids:
                intersection = expected_docs.intersection(retrieved_ids)
                precision = len(intersection) / len(retrieved_ids)
                recall = len(intersection) / len(expected_docs) if expected_docs else 0
                
                total_precision += precision
                total_recall += recall
        
        avg_precision = total_precision / total_queries
        avg_recall = total_recall / total_queries
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': round(avg_precision, 3),
            'recall': round(avg_recall, 3),
            'f1_score': round(f1_score, 3),
            'total_queries': total_queries
        }
    
    def evaluate_generation(self, 
                           test_qa_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate generation quality.
        
        Args:
            test_qa_pairs (List[Dict[str, str]]): Test Q&A pairs
            
        Returns:
            Dict[str, Any]: Generation evaluation metrics
        """
        if not test_qa_pairs:
            return {}
        
        results = []
        total_time = 0
        
        for qa_pair in test_qa_pairs:
            question = qa_pair.get('question', '')
            expected_answer = qa_pair.get('answer', '')
            
            start_time = time.time()
            result = self.rag_pipeline.query(question)
            query_time = time.time() - start_time
            
            generated_answer = result.get('answer', '')
            confidence = result.get('confidence', 0)
            
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'confidence': confidence,
                'query_time': query_time,
                'success': result.get('success', False)
            })
            
            total_time += query_time
        
        # Calculate basic metrics
        successful_queries = sum(1 for r in results if r['success'])
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_query_time = total_time / len(results)
        
        return {
            'total_queries': len(test_qa_pairs),
            'successful_queries': successful_queries,
            'success_rate': round(successful_queries / len(test_qa_pairs), 3),
            'average_confidence': round(avg_confidence, 3),
            'average_query_time': round(avg_query_time, 3),
            'detailed_results': results
        }

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize RAG pipeline
        rag = ArabicRAGPipeline()
        
        # Build knowledge base
        build_stats = rag.build_knowledge_base()
        print(f"Knowledge base build stats: {build_stats}")
        
        # Test query
        question = "ما هو الذكاء الاصطناعي؟"
        result = rag.query(question)
        print(f"Question: {question}")
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Confidence: {result.get('confidence', 0)}")
        
        # System stats
        stats = rag.get_system_stats()
        print(f"System stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed and you have sufficient resources.")