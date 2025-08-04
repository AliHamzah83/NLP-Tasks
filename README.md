# Arabic RAG System 🔍
## نظام الاسترجاع والتوليد العربي

A comprehensive Arabic Retrieval-Augmented Generation (RAG) system that combines document retrieval with Arabic text generation using state-of-the-art models.

## 🌟 Features

### Core Components
- **📚 Data Collection**: Configurable data directory with support for multiple file formats (TXT, JSON, CSV)
- **🔧 Text Preprocessing**: Arabic-specific preprocessing with noise removal, normalization, and tokenization using Farasa/CAMeL Tools
- **✂️ Chunking Strategies**: Fixed-size and sentence-based text splitting optimized for Arabic
- **🧠 Embedding Model**: Multilingual sentence embeddings using `paraphrase-multilingual-mpnet-base-v2`
- **🗄️ Vector Database**: ChromaDB integration for efficient similarity search
- **🤖 Language Model**: AraGPT2 for Arabic text generation
- **📊 Evaluation System**: Comprehensive metrics for retrieval and generation quality
- **🚀 Production Deployment**: FastAPI backend and Streamlit frontend

### Key Capabilities
- ✅ Arabic question answering with context retrieval
- ✅ Document management and knowledge base building
- ✅ Multi-strategy text chunking
- ✅ Real-time query processing
- ✅ Performance evaluation and metrics
- ✅ RESTful API with comprehensive endpoints
- ✅ Modern web interface with Arabic support
- ✅ Batch processing capabilities
- ✅ Export/import functionality

## 🏗️ Architecture

```
Arabic RAG System
├── Data Collection & Preprocessing
│   ├── Multi-format file loading (TXT, JSON, CSV)
│   ├── Arabic text normalization
│   ├── Tokenization (Farasa/CAMeL Tools)
│   └── Noise removal and cleaning
├── Text Chunking
│   ├── Fixed-size chunking
│   ├── Sentence-based chunking
│   └── Overlap management
├── Embedding & Retrieval
│   ├── Sentence Transformers (multilingual)
│   ├── ChromaDB vector storage
│   └── Similarity search
├── Generation
│   ├── AraGPT2 integration
│   ├── Context-aware generation
│   └── Confidence scoring
├── Evaluation
│   ├── Retrieval metrics (Precision, Recall, F1)
│   ├── Generation quality metrics
│   └── Performance analysis
└── Deployment
    ├── FastAPI backend
    ├── Streamlit frontend
    └── Production-ready APIs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)
- 8GB+ RAM
- 10GB+ disk space for models and data

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd arabic-rag-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download required NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Basic Usage

#### 1. Run the Demo
```bash
python demo.py
```

This will demonstrate all system capabilities including:
- Knowledge base building
- Arabic query processing
- Document management
- System evaluation

#### 2. Start the Backend API
```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

#### 3. Launch the Frontend
```bash
cd frontend
streamlit run streamlit_app.py
```

The web interface will be available at `http://localhost:8501`

## 📖 Detailed Usage

### Python API

#### Initialize the RAG System
```python
from arabic_rag import ArabicRAGPipeline

# Initialize with default settings
rag = ArabicRAGPipeline()

# Or customize configuration
rag = ArabicRAGPipeline(
    data_dir="my_data",
    embedding_model_name="paraphrase-multilingual-mpnet-base-v2",
    llm_model_name="aubmindlab/aragpt2-base",
    chunking_strategy="sentence_based",
    chunk_size=512,
    chunk_overlap=50
)
```

#### Build Knowledge Base
```python
# Build from data directory
build_stats = rag.build_knowledge_base()

# Force rebuild
build_stats = rag.build_knowledge_base(force_rebuild=True)
```

#### Query the System
```python
# Basic query
result = rag.query("ما هو الذكاء الاصطناعي؟")

# Advanced query with parameters
result = rag.query(
    question="كيف تعمل معالجة اللغات الطبيعية؟",
    top_k=5,
    max_contexts=3,
    similarity_threshold=0.5
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Processing time: {result['processing_time']}s")
```

#### Add Documents
```python
# Add custom documents
documents = [
    {
        "text": "النص العربي هنا...",
        "metadata": {"title": "عنوان الوثيقة", "author": "المؤلف"}
    }
]

success = rag.add_documents(documents)
```

#### Evaluation
```python
from arabic_rag import ArabicRAGEvaluator

evaluator = ArabicRAGEvaluator(rag)

# Create test dataset
test_questions = [
    "ما هو الذكاء الاصطناعي؟",
    "كيف تعمل الشبكات العصبية؟"
]

test_dataset = evaluator.create_test_dataset(test_questions)

# Run evaluation
metrics = evaluator.evaluate_end_to_end(test_dataset)

print(f"F1 Score: {metrics.f1_score}")
print(f"Answer Accuracy: {metrics.answer_accuracy}")
```

### REST API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Query
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "ما هو الذكاء الاصطناعي؟",
       "top_k": 5,
       "max_contexts": 3
     }'
```

#### Add Document
```bash
curl -X POST "http://localhost:8000/documents/add" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "النص العربي هنا...",
       "metadata": {"title": "عنوان الوثيقة"}
     }'
```

#### Build Knowledge Base
```bash
curl -X POST "http://localhost:8000/knowledge-base/build" \
     -H "Content-Type: application/json" \
     -d '{"force_rebuild": false}'
```

## ⚙️ Configuration

### System Configuration
```python
from arabic_rag import ArabicRAGPipeline

rag = ArabicRAGPipeline(
    # Data settings
    data_dir="data",                    # Data directory path
    db_path="chroma_db",               # Vector database path
    collection_name="arabic_documents", # Collection name
    
    # Model settings
    embedding_model_name="paraphrase-multilingual-mpnet-base-v2",
    llm_model_name="aubmindlab/aragpt2-base",
    
    # Chunking settings
    chunking_strategy="sentence_based", # or "fixed_size"
    chunk_size=512,                     # Characters per chunk
    chunk_overlap=50                    # Overlap between chunks
)
```

### Preprocessing Configuration
```python
from arabic_rag import PreprocessingConfig, ArabicTextPreprocessor

config = PreprocessingConfig(
    remove_diacritics=True,    # Remove Arabic diacritics
    normalize_alef=True,       # Normalize Alef variations
    normalize_teh_marbuta=True, # Normalize Teh Marbuta
    remove_punctuation=True,   # Remove punctuation
    remove_english=True,       # Remove English characters
    remove_numbers=False,      # Keep numbers
    remove_stopwords=True,     # Remove Arabic stopwords
    min_word_length=2,         # Minimum word length
    use_farasa=True,          # Use Farasa tokenizer
    use_camel=True            # Use CAMeL Tools
)

preprocessor = ArabicTextPreprocessor(config)
```

### Chunking Configuration
```python
from arabic_rag import ChunkingConfig, ArabicTextChunker

config = ChunkingConfig(
    chunk_size=512,           # Target chunk size
    chunk_overlap=50,         # Overlap between chunks
    min_chunk_size=100,       # Minimum chunk size
    max_chunk_size=1000,      # Maximum chunk size
    preserve_sentences=True,   # Preserve sentence boundaries
    strategy="sentence_based"  # Chunking strategy
)

chunker = ArabicTextChunker(config)
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Retrieval Metrics
- **Precision**: Fraction of retrieved documents that are relevant
- **Recall**: Fraction of relevant documents that are retrieved
- **F1-Score**: Harmonic mean of precision and recall
- **MAP (Mean Average Precision)**: Average precision across queries
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant document

### Generation Metrics
- **Answer Accuracy**: Exact match rate with reference answers
- **Answer Completeness**: Partial match rate based on word overlap
- **Semantic Similarity**: Cosine similarity between generated and reference answers
- **Confidence Score**: Model's confidence in generated answers

### Performance Metrics
- **Success Rate**: Percentage of successful query processing
- **Average Response Time**: Mean time to process queries
- **Throughput**: Queries processed per second

## 🗂️ Data Formats

### Supported Input Formats

#### Text Files (.txt)
```
الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات.
معالجة اللغات الطبيعية فرع مهم من الذكاء الاصطناعي.
```

#### JSON Files (.json)
```json
[
  {
    "text": "النص العربي هنا...",
    "title": "عنوان الوثيقة",
    "author": "المؤلف",
    "category": "تقنية"
  }
]
```

#### CSV Files (.csv)
```csv
text,title,author,category
"النص العربي الأول...","العنوان الأول","المؤلف الأول","تقنية"
"النص العربي الثاني...","العنوان الثاني","المؤلف الثاني","علوم"
```

## 🔧 Advanced Features

### Custom Embedding Models
```python
# Use different embedding models
rag = ArabicRAGPipeline(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Custom Language Models
```python
# Use different AraGPT2 variants
rag = ArabicRAGPipeline(
    llm_model_name="aubmindlab/aragpt2-medium"  # or aragpt2-large
)
```

### Batch Processing
```python
# Process multiple queries
questions = [
    "ما هو الذكاء الاصطناعي؟",
    "كيف تعمل الشبكات العصبية؟",
    "ما هي تطبيقات التعلم العميق؟"
]

results = rag.batch_query(questions)
```

### Export/Import
```python
# Export knowledge base
rag.export_knowledge_base("my_knowledge_base.json")

# Reset knowledge base
rag.reset_knowledge_base()
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Use CPU instead of GPU
rag = ArabicRAGPipeline()
# Models will automatically fall back to CPU
```

#### 2. Slow Performance
```python
# Reduce chunk size and batch size
rag = ArabicRAGPipeline(
    chunk_size=256,  # Smaller chunks
    chunk_overlap=25
)
```

#### 3. Poor Arabic Text Quality
```python
# Adjust preprocessing settings
from arabic_rag import PreprocessingConfig

config = PreprocessingConfig(
    remove_diacritics=False,  # Keep diacritics
    normalize_alef=True,      # Normalize Alef
    remove_stopwords=False    # Keep stopwords
)
```

#### 4. Low Retrieval Accuracy
```python
# Adjust similarity threshold
result = rag.query(
    question="سؤالك هنا",
    similarity_threshold=0.3  # Lower threshold
)
```

### Error Messages

- **"System not initialized"**: Run the initialization code first
- **"No documents found"**: Check data directory and file formats
- **"ChromaDB error"**: Ensure ChromaDB is properly installed
- **"Model loading failed"**: Check internet connection and disk space

## 📈 Performance Optimization

### Hardware Recommendations
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for large knowledge bases
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance
- **Storage**: SSD recommended for faster I/O

### Software Optimization
```python
# Use smaller models for faster inference
rag = ArabicRAGPipeline(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name="aubmindlab/aragpt2-base"  # Use base instead of large
)

# Optimize chunk size
rag = ArabicRAGPipeline(
    chunk_size=256,      # Smaller chunks for faster processing
    chunk_overlap=25     # Smaller overlap
)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 arabic_rag/
black arabic_rag/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **ChromaDB** for vector database capabilities
- **Sentence Transformers** for embedding models
- **AubmindLab** for AraGPT2 models
- **CAMeL Tools** and **Farasa** for Arabic NLP

## 📞 Support

For support and questions:
- 📧 Email: [your-email@domain.com]
- 💬 GitHub Issues: [Create an issue](../../issues)
- 📖 Documentation: [Wiki](../../wiki)

## 🗺️ Roadmap

### Upcoming Features
- [ ] Support for more Arabic dialects
- [ ] Integration with more embedding models
- [ ] Advanced evaluation metrics
- [ ] Multi-modal support (text + images)
- [ ] Real-time learning capabilities
- [ ] Enhanced web interface
- [ ] Mobile app support

### Version History
- **v1.0.0**: Initial release with core RAG functionality
- **v1.1.0**: Added evaluation system and metrics
- **v1.2.0**: Enhanced Arabic preprocessing
- **v1.3.0**: Added production deployment features

---

**Built with ❤️ for the Arabic NLP community**

نظام الاسترجاع والتوليد العربي - مصمم خصيصاً للغة العربية 🇸🇦