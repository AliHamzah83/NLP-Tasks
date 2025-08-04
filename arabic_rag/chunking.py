import re
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Try to import sentence splitting libraries
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available for sentence tokenization")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 512  # Size in characters for fixed-size chunking
    chunk_overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1000  # Maximum chunk size
    preserve_sentences: bool = True  # Try to preserve sentence boundaries
    strategy: str = "fixed_size"  # "fixed_size" or "sentence_based"

class TextChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, str]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text (str): Input text to chunk
            metadata (Dict): Optional metadata to include with chunks
            
        Returns:
            List[Dict[str, str]]: List of text chunks with metadata
        """
        pass

class FixedSizeChunker(TextChunker):
    """
    Fixed-size text chunker that splits text into chunks of specified size.
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize fixed-size chunker.
        
        Args:
            config (ChunkingConfig): Chunking configuration
        """
        self.config = config
        logger.info(f"Initialized FixedSizeChunker with size={config.chunk_size}, overlap={config.chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, str]]:
        """
        Split text into fixed-size chunks with optional overlap.
        
        Args:
            text (str): Input text to chunk
            metadata (Dict): Optional metadata to include with chunks
            
        Returns:
            List[Dict[str, str]]: List of text chunks with metadata
        """
        if not text or len(text) < self.config.min_chunk_size:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.config.chunk_size
            
            # If this is not the last chunk and we want to preserve sentences
            if end < len(text) and self.config.preserve_sentences:
                # Try to find a sentence boundary within the chunk
                chunk_text = text[start:end]
                
                # Look for Arabic sentence endings
                sentence_endings = ['؟', '!', '.', '؛', '\n']
                best_break = -1
                
                # Search backwards from the end for a sentence boundary
                for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 100), -1):
                    if chunk_text[i] in sentence_endings:
                        best_break = i + 1
                        break
                
                # If we found a good break point, use it
                if best_break > 0 and best_break > len(chunk_text) * 0.5:
                    end = start + best_break
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_size': len(chunk_text),
                    'chunking_strategy': 'fixed_size'
                }
                
                # Add original metadata if provided
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.config.chunk_overlap
            
            # Ensure we don't go backwards
            if start <= chunks[-1]['metadata']['start_pos'] if chunks else False:
                start = end
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks

class SentenceBasedChunker(TextChunker):
    """
    Sentence-based text chunker that groups sentences into chunks.
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize sentence-based chunker.
        
        Args:
            config (ChunkingConfig): Chunking configuration
        """
        self.config = config
        logger.info(f"Initialized SentenceBasedChunker with target_size={config.chunk_size}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK or basic regex.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        if NLTK_AVAILABLE:
            try:
                # Use NLTK sentence tokenizer
                sentences = sent_tokenize(text, language='arabic')
                return [s.strip() for s in sentences if s.strip()]
            except:
                pass
        
        # Fallback to regex-based sentence splitting
        # Arabic sentence endings
        sentence_pattern = r'[.!؟؛\n]+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, str]]:
        """
        Split text into sentence-based chunks.
        
        Args:
            text (str): Input text to chunk
            metadata (Dict): Optional metadata to include with chunks
            
        Returns:
            List[Dict[str, str]]: List of text chunks with metadata
        """
        if not text or len(text) < self.config.min_chunk_size:
            return []
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max size, finalize current chunk
            if (current_size + sentence_size > self.config.max_chunk_size and current_chunk) or \
               (current_size + sentence_size > self.config.chunk_size and current_chunk and current_size > self.config.min_chunk_size):
                
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk).strip()
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunk_metadata = {
                        'chunk_id': chunk_id,
                        'sentence_count': len(current_chunk),
                        'chunk_size': len(chunk_text),
                        'chunking_strategy': 'sentence_based'
                    }
                    
                    # Add original metadata if provided
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': chunk_metadata
                    })
                    
                    chunk_id += 1
                
                # Start new chunk with overlap if configured
                if self.config.chunk_overlap > 0 and current_chunk:
                    # Keep last few sentences for overlap
                    overlap_text = ' '.join(current_chunk)
                    if len(overlap_text) > self.config.chunk_overlap:
                        # Find good overlap point
                        overlap_sentences = []
                        overlap_size = 0
                        for sent in reversed(current_chunk):
                            if overlap_size + len(sent) <= self.config.chunk_overlap:
                                overlap_sentences.insert(0, sent)
                                overlap_size += len(sent)
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_size = overlap_size
                    else:
                        current_chunk = []
                        current_size = 0
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'sentence_count': len(current_chunk),
                    'chunk_size': len(chunk_text),
                    'chunking_strategy': 'sentence_based'
                }
                
                # Add original metadata if provided
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
        
        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks

class ArabicTextChunker:
    """
    Main chunking class that supports multiple chunking strategies.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize Arabic text chunker.
        
        Args:
            config (ChunkingConfig): Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Initialize chunkers
        self.fixed_size_chunker = FixedSizeChunker(self.config)
        self.sentence_based_chunker = SentenceBasedChunker(self.config)
        
        logger.info(f"Initialized ArabicTextChunker with strategy: {self.config.strategy}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, str]]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text (str): Input text to chunk
            metadata (Dict): Optional metadata to include with chunks
            
        Returns:
            List[Dict[str, str]]: List of text chunks with metadata
        """
        if self.config.strategy == "sentence_based":
            return self.sentence_based_chunker.chunk_text(text, metadata)
        else:
            return self.fixed_size_chunker.chunk_text(text, metadata)
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk a list of documents.
        
        Args:
            documents (List[Dict[str, str]]): List of documents with 'text' field
            
        Returns:
            List[Dict[str, str]]: List of text chunks with metadata
        """
        all_chunks = []
        
        for doc_id, document in enumerate(documents):
            text = document.get('text', '')
            if not text:
                continue
            
            # Prepare metadata
            doc_metadata = {
                'document_id': doc_id,
                'source': document.get('source', ''),
                'filename': document.get('filename', ''),
            }
            
            # Add any existing metadata
            if 'metadata' in document:
                doc_metadata.update(document['metadata'])
            
            # Chunk the document
            chunks = self.chunk_text(text, doc_metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, str]]) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks (List[Dict[str, str]]): List of chunks
            
        Returns:
            Dict[str, Union[int, float]]: Chunk statistics
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk['text']) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }
        
        # Strategy-specific stats
        if chunks[0]['metadata'].get('chunking_strategy') == 'sentence_based':
            sentence_counts = [chunk['metadata'].get('sentence_count', 0) for chunk in chunks]
            stats.update({
                'avg_sentences_per_chunk': sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
                'min_sentences_per_chunk': min(sentence_counts) if sentence_counts else 0,
                'max_sentences_per_chunk': max(sentence_counts) if sentence_counts else 0
            })
        
        return stats

if __name__ == "__main__":
    # Example usage
    config = ChunkingConfig(strategy="sentence_based", chunk_size=300)
    chunker = ArabicTextChunker(config)
    
    sample_text = """
    الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات المبرمجة للتفكير والتعلم مثل البشر.
    تعتبر معالجة اللغات الطبيعية فرعاً مهماً من فروع الذكاء الاصطناعي يهتم بفهم وتحليل اللغة البشرية.
    التعلم العميق هو تقنية تعتمد على الشبكات العصبية الاصطناعية لحل المشاكل المعقدة.
    البيانات الضخمة تلعب دوراً مهماً في تطوير أنظمة الذكاء الاصطناعي الحديثة.
    """
    
    chunks = chunker.chunk_text(sample_text)
    stats = chunker.get_chunk_statistics(chunks)
    
    print(f"Created {len(chunks)} chunks")
    print(f"Statistics: {stats}")