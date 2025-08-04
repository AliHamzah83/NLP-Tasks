import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicDataCollector:
    """
    Data collection module for Arabic RAG system.
    Supports various data formats and provides configurable data directory path.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data collector with configurable data directory.
        
        Args:
            data_dir (str): Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory set to: {self.data_dir.absolute()}")
    
    def load_text_files(self, file_pattern: str = "*.txt") -> List[Dict[str, str]]:
        """
        Load text files from data directory.
        
        Args:
            file_pattern (str): File pattern to match
            
        Returns:
            List[Dict[str, str]]: List of documents with metadata
        """
        documents = []
        text_files = list(self.data_dir.glob(file_pattern))
        
        if not text_files:
            logger.warning(f"No text files found matching pattern: {file_pattern}")
            return documents
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                documents.append({
                    'text': content,
                    'source': str(file_path),
                    'filename': file_path.name,
                    'size': len(content)
                })
                logger.info(f"Loaded: {file_path.name} ({len(content)} characters)")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_json_files(self, file_pattern: str = "*.json") -> List[Dict[str, str]]:
        """
        Load JSON files from data directory.
        
        Args:
            file_pattern (str): File pattern to match
            
        Returns:
            List[Dict[str, str]]: List of documents with metadata
        """
        documents = []
        json_files = list(self.data_dir.glob(file_pattern))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, dict) and 'text' in item:
                            documents.append({
                                'text': item['text'],
                                'source': str(file_path),
                                'filename': f"{file_path.stem}_{i}",
                                'metadata': {k: v for k, v in item.items() if k != 'text'}
                            })
                elif isinstance(data, dict) and 'text' in data:
                    documents.append({
                        'text': data['text'],
                        'source': str(file_path),
                        'filename': file_path.name,
                        'metadata': {k: v for k, v in data.items() if k != 'text'}
                    })
                    
                logger.info(f"Loaded JSON: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading JSON {file_path}: {e}")
        
        return documents
    
    def load_csv_files(self, text_column: str = "text", file_pattern: str = "*.csv") -> List[Dict[str, str]]:
        """
        Load CSV files from data directory.
        
        Args:
            text_column (str): Name of the text column
            file_pattern (str): File pattern to match
            
        Returns:
            List[Dict[str, str]]: List of documents with metadata
        """
        documents = []
        csv_files = list(self.data_dir.glob(file_pattern))
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                if text_column not in df.columns:
                    logger.error(f"Column '{text_column}' not found in {file_path}")
                    continue
                
                for idx, row in df.iterrows():
                    if pd.notna(row[text_column]):
                        documents.append({
                            'text': str(row[text_column]),
                            'source': str(file_path),
                            'filename': f"{file_path.stem}_{idx}",
                            'metadata': {col: row[col] for col in df.columns if col != text_column}
                        })
                
                logger.info(f"Loaded CSV: {file_path.name} ({len(df)} rows)")
                
            except Exception as e:
                logger.error(f"Error loading CSV {file_path}: {e}")
        
        return documents
    
    def load_all_data(self, text_column: str = "text") -> List[Dict[str, str]]:
        """
        Load all supported data formats from the data directory.
        
        Args:
            text_column (str): Name of the text column for CSV files
            
        Returns:
            List[Dict[str, str]]: Combined list of all documents
        """
        all_documents = []
        
        # Load text files
        all_documents.extend(self.load_text_files())
        
        # Load JSON files
        all_documents.extend(self.load_json_files())
        
        # Load CSV files
        all_documents.extend(self.load_csv_files(text_column))
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def create_sample_data(self):
        """
        Create sample Arabic data for testing purposes.
        """
        sample_texts = [
            "الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات المبرمجة للتفكير والتعلم مثل البشر.",
            "تعتبر معالجة اللغات الطبيعية فرعاً مهماً من فروع الذكاء الاصطناعي يهتم بفهم وتحليل اللغة البشرية.",
            "التعلم العميق هو تقنية تعتمد على الشبكات العصبية الاصطناعية لحل المشاكل المعقدة.",
            "البيانات الضخمة تلعب دوراً مهماً في تطوير أنظمة الذكاء الاصطناعي الحديثة.",
            "الحوسبة السحابية توفر البنية التحتية اللازمة لتشغيل تطبيقات الذكاء الاصطناعي على نطاق واسع."
        ]
        
        sample_file = self.data_dir / "sample_arabic_data.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(sample_texts))
        
        logger.info(f"Created sample data file: {sample_file}")
        
        # Create sample JSON data
        json_data = [
            {"text": text, "category": "AI", "language": "Arabic"} 
            for text in sample_texts
        ]
        
        json_file = self.data_dir / "sample_arabic_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created sample JSON file: {json_file}")

if __name__ == "__main__":
    # Example usage
    collector = ArabicDataCollector("data")
    collector.create_sample_data()
    documents = collector.load_all_data()
    print(f"Loaded {len(documents)} documents")