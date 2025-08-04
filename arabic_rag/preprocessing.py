import re
import string
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

# Try to import Arabic NLP libraries
try:
    from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
    from camel_tools.utils.dediac import dediac_ar
    from camel_tools.tokenizers.word import simple_word_tokenize
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    logging.warning("CAMeL Tools not available. Using basic preprocessing.")

try:
    from farasa.segmenter import FarasaSegmenter
    FARASA_AVAILABLE = True
except ImportError:
    FARASA_AVAILABLE = False
    logging.warning("Farasa not available. Using basic tokenization.")

import nltk
from nltk.corpus import stopwords

# Download Arabic stopwords if not available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for Arabic text preprocessing."""
    remove_diacritics: bool = True
    normalize_alef: bool = True
    normalize_teh_marbuta: bool = True
    remove_punctuation: bool = True
    remove_english: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    min_word_length: int = 2
    use_farasa: bool = True
    use_camel: bool = True

class ArabicTextPreprocessor:
    """
    Arabic text preprocessing with noise removal, normalization, and tokenization.
    Supports both Farasa and CAMeL Tools.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config (PreprocessingConfig): Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
        # Initialize Farasa segmenter if available
        self.farasa_segmenter = None
        if self.config.use_farasa and FARASA_AVAILABLE:
            try:
                self.farasa_segmenter = FarasaSegmenter(interactive=True)
                logger.info("Farasa segmenter initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Farasa: {e}")
        
        # Load Arabic stopwords
        try:
            self.arabic_stopwords = set(stopwords.words('arabic'))
        except:
            # Fallback to basic Arabic stopwords
            self.arabic_stopwords = {
                'في', 'من', 'إلى', 'على', 'أن', 'هذا', 'هذه', 'التي', 'الذي', 
                'كان', 'كانت', 'يكون', 'تكون', 'له', 'لها', 'هو', 'هي', 'أو',
                'لا', 'ما', 'كل', 'بعض', 'عند', 'عندما', 'حيث', 'بين', 'تحت',
                'فوق', 'أمام', 'خلف', 'يمين', 'شمال', 'داخل', 'خارج'
            }
        
        logger.info("Arabic text preprocessor initialized")
    
    def remove_noise(self, text: str) -> str:
        """
        Remove noise from Arabic text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove English characters if specified
        if self.config.remove_english:
            text = re.sub(r'[a-zA-Z]', '', text)
        
        # Remove numbers if specified
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text using CAMeL Tools or basic normalization.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if CAMEL_AVAILABLE and self.config.use_camel:
            # Remove diacritics
            if self.config.remove_diacritics:
                text = dediac_ar(text)
            
            # Normalize Alef variations
            if self.config.normalize_alef:
                text = normalize_alef_ar(text)
                text = normalize_alef_maksura_ar(text)
            
            # Normalize Teh Marbuta
            if self.config.normalize_teh_marbuta:
                text = normalize_teh_marbuta_ar(text)
        else:
            # Basic normalization
            if self.config.remove_diacritics:
                # Remove Arabic diacritics
                arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
                text = arabic_diacritics.sub('', text)
            
            if self.config.normalize_alef:
                # Normalize Alef variations
                text = re.sub(r'[إأآا]', 'ا', text)
                text = re.sub(r'ى', 'ي', text)
            
            if self.config.normalize_teh_marbuta:
                # Normalize Teh Marbuta
                text = re.sub(r'ة', 'ه', text)
        
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without punctuation
        """
        if self.config.remove_punctuation:
            # Arabic punctuation
            arabic_punctuation = '،؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؞؟؀؁؂؃؄؅؆؇؈؉؊؋،؍؎؏'
            # Combine with English punctuation
            all_punctuation = string.punctuation + arabic_punctuation
            
            # Remove punctuation
            translator = str.maketrans('', '', all_punctuation)
            text = text.translate(translator)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize Arabic text using Farasa, CAMeL Tools, or basic tokenization.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        tokens = []
        
        if self.farasa_segmenter and self.config.use_farasa:
            try:
                # Use Farasa for tokenization
                segmented = self.farasa_segmenter.segment(text)
                tokens = segmented.split()
            except Exception as e:
                logger.warning(f"Farasa tokenization failed: {e}, falling back to basic tokenization")
                tokens = self._basic_tokenize(text)
        elif CAMEL_AVAILABLE and self.config.use_camel:
            try:
                # Use CAMeL Tools for tokenization
                tokens = simple_word_tokenize(text)
            except Exception as e:
                logger.warning(f"CAMeL tokenization failed: {e}, falling back to basic tokenization")
                tokens = self._basic_tokenize(text)
        else:
            tokens = self._basic_tokenize(text)
        
        return tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """
        Basic tokenization by splitting on whitespace.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove Arabic stopwords from tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Filtered tokens
        """
        if self.config.remove_stopwords:
            tokens = [token for token in tokens if token not in self.arabic_stopwords]
        
        return tokens
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on length and other criteria.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Filtered tokens
        """
        filtered_tokens = []
        
        for token in tokens:
            # Remove short tokens
            if len(token) < self.config.min_word_length:
                continue
            
            # Remove empty tokens
            if not token.strip():
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Step 1: Remove noise
        text = self.remove_noise(text)
        
        # Step 2: Normalize Arabic text
        text = self.normalize_arabic(text)
        
        # Step 3: Remove punctuation
        text = self.remove_punctuation(text)
        
        # Step 4: Tokenize
        tokens = self.tokenize_text(text)
        
        # Step 5: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 6: Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Join tokens back to text
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Preprocess a list of documents.
        
        Args:
            documents (List[Dict[str, str]]): List of documents with 'text' field
            
        Returns:
            List[Dict[str, str]]: List of preprocessed documents
        """
        preprocessed_docs = []
        
        for doc in documents:
            try:
                original_text = doc.get('text', '')
                preprocessed_text = self.preprocess_text(original_text)
                
                # Create new document with preprocessed text
                new_doc = doc.copy()
                new_doc['text'] = preprocessed_text
                new_doc['original_text'] = original_text
                new_doc['preprocessing_applied'] = True
                
                preprocessed_docs.append(new_doc)
                
            except Exception as e:
                logger.error(f"Error preprocessing document: {e}")
                # Keep original document if preprocessing fails
                preprocessed_docs.append(doc)
        
        logger.info(f"Preprocessed {len(preprocessed_docs)} documents")
        return preprocessed_docs

if __name__ == "__main__":
    # Example usage
    config = PreprocessingConfig()
    preprocessor = ArabicTextPreprocessor(config)
    
    sample_text = "هذا نص تجريبي باللغة العربية، يحتوي على علامات ترقيم وأرقام 123."
    processed = preprocessor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")