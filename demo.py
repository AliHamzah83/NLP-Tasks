#!/usr/bin/env python3
"""
Arabic RAG System Demo Script

This script demonstrates the complete functionality of the Arabic RAG system
including data processing, knowledge base building, querying, and evaluation.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append('.')

from arabic_rag import (
    ArabicRAGPipeline, 
    ArabicRAGEvaluator, 
    EvaluationMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step: str):
    """Print a formatted step."""
    print(f"\n🔄 {step}")
    print("-" * 50)

def demo_basic_functionality():
    """Demonstrate basic RAG functionality."""
    print_header("Arabic RAG System Demo - Basic Functionality")
    
    try:
        # Step 1: Initialize the RAG pipeline
        print_step("Step 1: Initializing Arabic RAG Pipeline")
        rag = ArabicRAGPipeline(
            data_dir="data",
            db_path="demo_chroma_db",
            collection_name="arabic_demo",
            embedding_model_name="paraphrase-multilingual-mpnet-base-v2",
            llm_model_name="aubmindlab/aragpt2-base",
            chunking_strategy="sentence_based",
            chunk_size=512,
            chunk_overlap=50
        )
        print("✅ RAG pipeline initialized successfully!")
        
        # Step 2: Build knowledge base
        print_step("Step 2: Building Knowledge Base")
        build_stats = rag.build_knowledge_base(force_rebuild=True)
        
        if build_stats.get('success', False):
            print(f"✅ Knowledge base built successfully!")
            print(f"   📊 Total documents: {build_stats.get('total_documents', 0)}")
            print(f"   📄 Total chunks: {build_stats.get('total_chunks', 0)}")
            print(f"   ⏱️  Build time: {build_stats.get('build_time_seconds', 0):.2f} seconds")
            print(f"   🧠 Embedding model: {build_stats.get('embedding_model', 'N/A')}")
            print(f"   ✂️  Chunking strategy: {build_stats.get('chunking_strategy', 'N/A')}")
        else:
            print(f"❌ Knowledge base build failed: {build_stats.get('error', 'Unknown error')}")
            return
        
        # Step 3: Test queries
        print_step("Step 3: Testing Arabic Queries")
        
        test_questions = [
            "ما هو الذكاء الاصطناعي؟",
            "كيف تعمل معالجة اللغات الطبيعية؟",
            "ما هي تطبيقات التعلم العميق؟",
            "ما هي البيانات الضخمة؟",
            "كيف تعمل الحوسبة السحابية؟"
        ]
        
        results = []
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Query {i}: {question}")
            
            start_time = time.time()
            result = rag.query(
                question=question,
                top_k=5,
                max_contexts=3,
                similarity_threshold=0.3
            )
            query_time = time.time() - start_time
            
            if result.get('success', False):
                print(f"✅ Answer: {result.get('answer', 'No answer')}")
                print(f"   🎯 Confidence: {result.get('confidence', 0):.2%}")
                print(f"   ⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
                print(f"   📚 Retrieved docs: {len(result.get('retrieved_docs', []))}")
                
                # Show top retrieved document
                retrieved_docs = result.get('retrieved_docs', [])
                if retrieved_docs:
                    top_doc = retrieved_docs[0]
                    print(f"   🔝 Top document similarity: {top_doc.get('similarity_score', 0):.3f}")
                    print(f"   📄 Top document preview: {top_doc.get('text', '')[:100]}...")
                
                results.append(result)
            else:
                print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        
        # Step 4: System statistics
        print_step("Step 4: System Statistics")
        stats = rag.get_system_stats()
        
        print("📊 Vector Database:")
        vector_stats = stats.get('vector_database', {})
        print(f"   📄 Total documents: {vector_stats.get('total_documents', 0)}")
        print(f"   🗂️  Collection: {vector_stats.get('collection_name', 'N/A')}")
        
        print("\n🧠 Embedding Model:")
        embedding_stats = stats.get('embedding_model', {})
        print(f"   🏷️  Model: {embedding_stats.get('model_name', 'N/A')}")
        print(f"   📐 Dimensions: {embedding_stats.get('embedding_dim', 0)}")
        print(f"   💻 Device: {embedding_stats.get('device', 'N/A')}")
        
        print("\n🤖 Language Model:")
        llm_stats = stats.get('language_model', {})
        print(f"   🏷️  Model: {llm_stats.get('model_name', 'N/A')}")
        print(f"   💻 Device: {llm_stats.get('device', 'N/A')}")
        print(f"   📏 Max length: {llm_stats.get('max_length', 0)}")
        
        # Step 5: Performance summary
        print_step("Step 5: Performance Summary")
        if results:
            avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
            avg_time = sum(r.get('processing_time', 0) for r in results) / len(results)
            success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
            
            print(f"📈 Average confidence: {avg_confidence:.2%}")
            print(f"⏱️  Average processing time: {avg_time:.2f}s")
            print(f"✅ Success rate: {success_rate:.2%}")
        
        print_step("Demo Completed Successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

def demo_evaluation():
    """Demonstrate evaluation functionality."""
    print_header("Arabic RAG System Demo - Evaluation")
    
    try:
        # Initialize system
        print_step("Initializing System for Evaluation")
        rag = ArabicRAGPipeline(
            data_dir="data",
            db_path="demo_chroma_db",
            collection_name="arabic_demo"
        )
        
        # Build knowledge base if needed
        stats = rag.vector_db.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            print("Building knowledge base for evaluation...")
            rag.build_knowledge_base()
        
        # Initialize evaluator
        evaluator = ArabicRAGEvaluator(rag)
        
        # Create test dataset
        print_step("Creating Test Dataset")
        test_questions = [
            "ما هو الذكاء الاصطناعي؟",
            "كيف تعمل معالجة اللغات الطبيعية؟",
            "ما هي تطبيقات التعلم العميق؟",
            "ما هي البيانات الضخمة وما أهميتها؟",
            "كيف تساعد الحوسبة السحابية في الذكاء الاصطناعي؟"
        ]
        
        test_dataset = evaluator.create_test_dataset(
            questions=test_questions,
            generate_answers=True
        )
        print(f"✅ Created test dataset with {len(test_dataset)} items")
        
        # Run evaluation
        print_step("Running Comprehensive Evaluation")
        metrics = evaluator.evaluate_end_to_end(
            test_dataset=test_dataset,
            save_results=True,
            output_path="demo_evaluation_results.json"
        )
        
        # Display results
        print_step("Evaluation Results")
        print("🎯 Retrieval Metrics:")
        print(f"   Precision: {metrics.precision:.4f}")
        print(f"   Recall: {metrics.recall:.4f}")
        print(f"   F1-Score: {metrics.f1_score:.4f}")
        
        print("\n📝 Generation Metrics:")
        print(f"   Answer Accuracy: {metrics.answer_accuracy:.4f}")
        print(f"   Answer Completeness: {metrics.answer_completeness:.4f}")
        print(f"   Semantic Similarity: {metrics.semantic_similarity:.4f}")
        
        print("\n⚡ Performance Metrics:")
        print(f"   Success Rate: {metrics.success_rate:.4f}")
        print(f"   Average Response Time: {metrics.average_response_time:.4f}s")
        print(f"   Confidence Score: {metrics.confidence_score:.4f}")
        
        # Generate report
        print_step("Generating Evaluation Report")
        report = evaluator.generate_evaluation_report(
            metrics=metrics,
            output_path="demo_evaluation_report.txt"
        )
        print("✅ Evaluation report saved to 'demo_evaluation_report.txt'")
        
        print_step("Evaluation Demo Completed! 📊")
        
    except Exception as e:
        logger.error(f"Evaluation demo failed: {e}")
        print(f"❌ Evaluation demo failed: {e}")

def demo_document_management():
    """Demonstrate document management functionality."""
    print_header("Arabic RAG System Demo - Document Management")
    
    try:
        # Initialize system
        rag = ArabicRAGPipeline(
            data_dir="data",
            db_path="demo_chroma_db",
            collection_name="arabic_demo"
        )
        
        # Add custom documents
        print_step("Adding Custom Documents")
        
        custom_documents = [
            {
                "text": "الذكاء الاصطناعي هو مجال في علوم الحاسوب يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً. يشمل هذا المجال التعلم الآلي، ومعالجة اللغات الطبيعية، والرؤية الحاسوبية، والروبوتات.",
                "metadata": {"title": "مقدمة في الذكاء الاصطناعي", "category": "تعليمي", "author": "نظام RAG"}
            },
            {
                "text": "التعلم العميق هو فرع من التعلم الآلي يستخدم الشبكات العصبية الاصطناعية ذات الطبقات المتعددة لتحليل البيانات. يُستخدم في تطبيقات متنوعة مثل التعرف على الصور والكلام والترجمة الآلية.",
                "metadata": {"title": "التعلم العميق", "category": "تقنية", "author": "نظام RAG"}
            },
            {
                "text": "البيانات الضخمة تشير إلى مجموعات البيانات الكبيرة والمعقدة التي تتطلب أدوات وتقنيات خاصة لمعالجتها وتحليلها. تتميز بالحجم الكبير، والسرعة العالية، والتنوع في أنواع البيانات.",
                "metadata": {"title": "البيانات الضخمة", "category": "بيانات", "author": "نظام RAG"}
            }
        ]
        
        success = rag.add_documents(custom_documents)
        if success:
            print(f"✅ Added {len(custom_documents)} custom documents")
        else:
            print("❌ Failed to add custom documents")
        
        # Test queries on new documents
        print_step("Testing Queries on New Documents")
        
        test_query = "ما هي خصائص البيانات الضخمة؟"
        result = rag.query(test_query, top_k=3)
        
        if result.get('success'):
            print(f"📝 Query: {test_query}")
            print(f"✅ Answer: {result.get('answer', 'No answer')}")
            print(f"🎯 Confidence: {result.get('confidence', 0):.2%}")
            
            # Show retrieved documents
            retrieved_docs = result.get('retrieved_docs', [])
            print(f"📚 Retrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Unknown')
                similarity = doc.get('similarity_score', 0)
                print(f"   {i+1}. {title} (similarity: {similarity:.3f})")
        
        # Export knowledge base
        print_step("Exporting Knowledge Base")
        export_success = rag.export_knowledge_base("demo_knowledge_base_export.json")
        if export_success:
            print("✅ Knowledge base exported successfully")
        else:
            print("❌ Failed to export knowledge base")
        
        # Get final statistics
        print_step("Final System Statistics")
        final_stats = rag.get_system_stats()
        vector_stats = final_stats.get('vector_database', {})
        print(f"📊 Total documents in knowledge base: {vector_stats.get('total_documents', 0)}")
        
        print_step("Document Management Demo Completed! 📄")
        
    except Exception as e:
        logger.error(f"Document management demo failed: {e}")
        print(f"❌ Document management demo failed: {e}")

def main():
    """Main demo function."""
    print_header("🔍 Arabic RAG System - Complete Demo")
    print("Welcome to the Arabic Retrieval-Augmented Generation System Demo!")
    print("This demo will showcase all major features of the system.")
    
    # Check if required directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    
    try:
        # Demo 1: Basic functionality
        demo_basic_functionality()
        
        # Demo 2: Document management
        demo_document_management()
        
        # Demo 3: Evaluation
        demo_evaluation()
        
        # Final summary
        print_header("🎉 Demo Summary")
        print("✅ All demos completed successfully!")
        print("\nWhat was demonstrated:")
        print("1. 🏗️  RAG pipeline initialization and configuration")
        print("2. 📚 Knowledge base building from sample data")
        print("3. 🔍 Arabic question answering with retrieval")
        print("4. 📄 Document management (add, export)")
        print("5. 📊 System evaluation and metrics")
        print("6. 📈 Performance analysis and reporting")
        
        print("\nGenerated files:")
        print("- demo_evaluation_results.json (evaluation metrics)")
        print("- demo_evaluation_report.txt (human-readable report)")
        print("- demo_knowledge_base_export.json (knowledge base export)")
        
        print("\nNext steps:")
        print("1. 🚀 Run the backend: python backend/main.py")
        print("2. 🌐 Run the frontend: streamlit run frontend/streamlit_app.py")
        print("3. 📖 Check the README.md for detailed instructions")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()