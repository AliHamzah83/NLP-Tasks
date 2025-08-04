from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import asyncio
import json
import os
from pathlib import Path
import time
import uuid

# Import our Arabic RAG system
import sys
sys.path.append('..')
from arabic_rag import ArabicRAGPipeline, ArabicRAGEvaluator, EvaluationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Arabic RAG System API",
    description="RESTful API for Arabic Retrieval-Augmented Generation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_pipeline: Optional[ArabicRAGPipeline] = None
evaluator: Optional[ArabicRAGEvaluator] = None
system_status = {"initialized": False, "last_update": None}

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    max_contexts: int = 3
    similarity_threshold: float = 0.5

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    processing_time: float
    success: bool
    retrieved_docs: List[Dict[str, Any]]
    error: Optional[str] = None

class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class BuildKnowledgeBaseRequest(BaseModel):
    force_rebuild: bool = False
    text_column: str = "text"

class EvaluationRequest(BaseModel):
    test_questions: List[str]
    generate_answers: bool = True

class SystemConfig(BaseModel):
    data_dir: str = "data"
    db_path: str = "chroma_db"
    collection_name: str = "arabic_documents"
    embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    llm_model_name: str = "aubmindlab/aragpt2-base"
    chunking_strategy: str = "sentence_based"
    chunk_size: int = 512
    chunk_overlap: int = 50

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_pipeline, evaluator, system_status
    
    try:
        logger.info("Initializing Arabic RAG System...")
        
        # Initialize RAG pipeline with default configuration
        rag_pipeline = ArabicRAGPipeline()
        
        # Initialize evaluator
        evaluator = ArabicRAGEvaluator(rag_pipeline)
        
        # Update system status
        system_status["initialized"] = True
        system_status["last_update"] = time.time()
        
        logger.info("Arabic RAG System initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        system_status["initialized"] = False
        system_status["error"] = str(e)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if system_status["initialized"] else "unhealthy",
        "timestamp": time.time(),
        "system_initialized": system_status["initialized"]
    }

# System status endpoint
@app.get("/status")
async def get_system_status():
    """Get detailed system status."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        stats = rag_pipeline.get_system_stats()
        return {
            "initialized": system_status["initialized"],
            "last_update": system_status["last_update"],
            "system_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    """Query the RAG system with a question."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"Processing query: {request.question}")
        
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            max_contexts=request.max_contexts,
            similarity_threshold=request.similarity_threshold
        )
        
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            confidence=result.get("confidence", 0.0),
            processing_time=result.get("processing_time", 0.0),
            success=result.get("success", False),
            retrieved_docs=result.get("retrieved_docs", []),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

# Batch query endpoint
@app.post("/query/batch")
async def batch_query(questions: List[str], top_k: int = 5, max_contexts: int = 3):
    """Process multiple questions in batch."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        results = rag_pipeline.batch_query(
            questions=questions,
            top_k=top_k,
            max_contexts=max_contexts
        )
        
        return {"results": results, "total_questions": len(questions)}
        
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch query failed: {e}")

# Build knowledge base endpoint
@app.post("/knowledge-base/build")
async def build_knowledge_base(request: BuildKnowledgeBaseRequest, background_tasks: BackgroundTasks):
    """Build or rebuild the knowledge base."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    def build_kb():
        try:
            result = rag_pipeline.build_knowledge_base(
                force_rebuild=request.force_rebuild,
                text_column=request.text_column
            )
            system_status["last_update"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Knowledge base build failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Run build in background
    background_tasks.add_task(build_kb)
    
    return {"message": "Knowledge base build started", "status": "building"}

# Add document endpoint
@app.post("/documents/add")
async def add_document(request: DocumentRequest):
    """Add a new document to the knowledge base."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        document = {
            "text": request.text,
            "metadata": request.metadata or {}
        }
        
        success = rag_pipeline.add_documents([document])
        
        if success:
            system_status["last_update"] = time.time()
            return {"message": "Document added successfully", "success": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to add document")
            
    except Exception as e:
        logger.error(f"Add document failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {e}")

# Upload file endpoint
@app.post("/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file based on type
        if file.filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            document = {
                "text": text,
                "metadata": {"filename": file.filename, "source": "upload"}
            }
            
            success = rag_pipeline.add_documents([document])
            
            if success:
                system_status["last_update"] = time.time()
                return {"message": f"File {file.filename} processed successfully", "success": True}
            else:
                raise HTTPException(status_code=500, detail="Failed to process file")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
            
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

# Evaluation endpoint
@app.post("/evaluate")
async def evaluate_system(request: EvaluationRequest):
    """Evaluate the RAG system performance."""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        # Create test dataset
        test_dataset = evaluator.create_test_dataset(
            questions=request.test_questions,
            generate_answers=request.generate_answers
        )
        
        # Run evaluation
        metrics = evaluator.evaluate_end_to_end(test_dataset)
        
        return {
            "evaluation_completed": True,
            "metrics": metrics.to_dict(),
            "test_dataset_size": len(test_dataset)
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

# Export knowledge base endpoint
@app.get("/knowledge-base/export")
async def export_knowledge_base():
    """Export the knowledge base to a file."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        export_path = f"exports/knowledge_base_export_{int(time.time())}.json"
        os.makedirs("exports", exist_ok=True)
        
        success = rag_pipeline.export_knowledge_base(export_path)
        
        if success:
            return FileResponse(
                path=export_path,
                filename=f"knowledge_base_export.json",
                media_type="application/json"
            )
        else:
            raise HTTPException(status_code=500, detail="Export failed")
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

# Reset knowledge base endpoint
@app.post("/knowledge-base/reset")
async def reset_knowledge_base():
    """Reset (clear) the knowledge base."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        success = rag_pipeline.reset_knowledge_base()
        
        if success:
            system_status["last_update"] = time.time()
            return {"message": "Knowledge base reset successfully", "success": True}
        else:
            raise HTTPException(status_code=500, detail="Reset failed")
            
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

# Configuration endpoint
@app.post("/configure")
async def configure_system(config: SystemConfig):
    """Reconfigure the RAG system."""
    global rag_pipeline, evaluator, system_status
    
    try:
        logger.info("Reconfiguring RAG system...")
        
        # Reinitialize with new configuration
        rag_pipeline = ArabicRAGPipeline(
            data_dir=config.data_dir,
            db_path=config.db_path,
            collection_name=config.collection_name,
            embedding_model_name=config.embedding_model_name,
            llm_model_name=config.llm_model_name,
            chunking_strategy=config.chunking_strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        evaluator = ArabicRAGEvaluator(rag_pipeline)
        
        system_status["initialized"] = True
        system_status["last_update"] = time.time()
        
        return {"message": "System reconfigured successfully", "success": True}
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        system_status["initialized"] = False
        raise HTTPException(status_code=500, detail=f"Configuration failed: {e}")

# Get available models endpoint
@app.get("/models")
async def get_available_models():
    """Get list of available models."""
    return {
        "embedding_models": [
            "paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ],
        "llm_models": [
            "aubmindlab/aragpt2-base",
            "aubmindlab/aragpt2-medium",
            "aubmindlab/aragpt2-large"
        ],
        "chunking_strategies": [
            "fixed_size",
            "sentence_based"
        ]
    }

# Websocket endpoint for real-time updates (optional)
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                if rag_pipeline:
                    result = rag_pipeline.query(message["question"])
                    await websocket.send_text(json.dumps(result))
                else:
                    await websocket.send_text(json.dumps({"error": "System not initialized"}))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )