import logging
import time
import json
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

# Try to import additional evaluation libraries
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available for advanced metrics")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available for plotting")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Retrieval metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map_score: float = 0.0  # Mean Average Precision
    mrr_score: float = 0.0  # Mean Reciprocal Rank
    
    # Generation metrics
    bleu_score: float = 0.0
    rouge_l: float = 0.0
    semantic_similarity: float = 0.0
    answer_relevance: float = 0.0
    
    # System metrics
    average_response_time: float = 0.0
    success_rate: float = 0.0
    confidence_score: float = 0.0
    
    # Quality metrics
    answer_completeness: float = 0.0
    answer_accuracy: float = 0.0
    factual_consistency: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'map_score': self.map_score,
            'mrr_score': self.mrr_score,
            'bleu_score': self.bleu_score,
            'rouge_l': self.rouge_l,
            'semantic_similarity': self.semantic_similarity,
            'answer_relevance': self.answer_relevance,
            'average_response_time': self.average_response_time,
            'success_rate': self.success_rate,
            'confidence_score': self.confidence_score,
            'answer_completeness': self.answer_completeness,
            'answer_accuracy': self.answer_accuracy,
            'factual_consistency': self.factual_consistency
        }

class ArabicRAGEvaluator:
    """
    Comprehensive evaluator for Arabic RAG system performance.
    Includes retrieval, generation, and system-level metrics.
    """
    
    def __init__(self, rag_pipeline):
        """
        Initialize evaluator.
        
        Args:
            rag_pipeline: Arabic RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.evaluation_history = []
        logger.info("Arabic RAG Evaluator initialized")
    
    def evaluate_retrieval(self, 
                          test_queries: List[Dict[str, Any]],
                          top_k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval performance with multiple metrics.
        
        Args:
            test_queries (List[Dict[str, Any]]): Test queries with ground truth
            top_k (int): Number of documents to retrieve
            
        Returns:
            Dict[str, float]: Retrieval evaluation metrics
        """
        if not test_queries:
            return {}
        
        logger.info(f"Evaluating retrieval performance on {len(test_queries)} queries")
        
        precision_scores = []
        recall_scores = []
        ap_scores = []  # Average Precision scores
        rr_scores = []  # Reciprocal Rank scores
        
        for query_data in test_queries:
            question = query_data.get('question', '')
            relevant_doc_ids = set(query_data.get('relevant_docs', []))
            
            if not relevant_doc_ids:
                continue
            
            # Retrieve documents
            retrieved_docs = self.rag_pipeline.vector_db.search_by_text(
                query_text=question,
                embedding_model=self.rag_pipeline.embedding_model,
                top_k=top_k
            )
            
            retrieved_ids = [
                doc.get('metadata', {}).get('document_id', f"doc_{i}") 
                for i, doc in enumerate(retrieved_docs)
            ]
            
            # Calculate metrics for this query
            precision, recall, ap, rr = self._calculate_retrieval_metrics(
                retrieved_ids, relevant_doc_ids
            )
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            ap_scores.append(ap)
            rr_scores.append(rr)
        
        # Calculate aggregate metrics
        avg_precision = np.mean(precision_scores) if precision_scores else 0.0
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0
        f1_score = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0.0
        map_score = np.mean(ap_scores) if ap_scores else 0.0
        mrr_score = np.mean(rr_scores) if rr_scores else 0.0
        
        results = {
            'precision': round(avg_precision, 4),
            'recall': round(avg_recall, 4),
            'f1_score': round(f1_score, 4),
            'map_score': round(map_score, 4),
            'mrr_score': round(mrr_score, 4),
            'total_queries': len(test_queries),
            'evaluated_queries': len(precision_scores)
        }
        
        logger.info(f"Retrieval evaluation completed: {results}")
        return results
    
    def _calculate_retrieval_metrics(self, 
                                   retrieved_ids: List[str], 
                                   relevant_ids: set) -> Tuple[float, float, float, float]:
        """Calculate retrieval metrics for a single query."""
        if not retrieved_ids:
            return 0.0, 0.0, 0.0, 0.0
        
        # Precision and Recall
        retrieved_set = set(retrieved_ids)
        intersection = relevant_ids.intersection(retrieved_set)
        
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(intersection) / len(relevant_ids) if relevant_ids else 0.0
        
        # Average Precision (AP)
        ap = 0.0
        relevant_found = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                ap += precision_at_i
        
        ap = ap / len(relevant_ids) if relevant_ids else 0.0
        
        # Reciprocal Rank (RR)
        rr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                rr = 1.0 / (i + 1)
                break
        
        return precision, recall, ap, rr
    
    def evaluate_generation(self, 
                           test_qa_pairs: List[Dict[str, str]],
                           include_advanced_metrics: bool = True) -> Dict[str, Any]:
        """
        Evaluate text generation quality.
        
        Args:
            test_qa_pairs (List[Dict[str, str]]): Test Q&A pairs
            include_advanced_metrics (bool): Whether to include advanced NLP metrics
            
        Returns:
            Dict[str, Any]: Generation evaluation results
        """
        if not test_qa_pairs:
            return {}
        
        logger.info(f"Evaluating generation quality on {len(test_qa_pairs)} Q&A pairs")
        
        results = []
        response_times = []
        success_count = 0
        confidence_scores = []
        
        # Basic metrics
        exact_matches = 0
        partial_matches = 0
        semantic_similarities = []
        answer_lengths = []
        
        for qa_pair in test_qa_pairs:
            question = qa_pair.get('question', '')
            expected_answer = qa_pair.get('answer', '')
            
            # Generate answer
            start_time = time.time()
            result = self.rag_pipeline.query(question)
            response_time = time.time() - start_time
            
            generated_answer = result.get('answer', '')
            confidence = result.get('confidence', 0.0)
            success = result.get('success', False)
            
            # Track metrics
            response_times.append(response_time)
            confidence_scores.append(confidence)
            answer_lengths.append(len(generated_answer.split()))
            
            if success:
                success_count += 1
            
            # Text similarity metrics
            exact_match = self._calculate_exact_match(expected_answer, generated_answer)
            partial_match = self._calculate_partial_match(expected_answer, generated_answer)
            
            if exact_match:
                exact_matches += 1
            if partial_match:
                partial_matches += 1
            
            # Semantic similarity
            if include_advanced_metrics:
                semantic_sim = self._calculate_semantic_similarity(expected_answer, generated_answer)
                semantic_similarities.append(semantic_sim)
            
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'confidence': confidence,
                'response_time': response_time,
                'success': success,
                'exact_match': exact_match,
                'partial_match': partial_match,
                'semantic_similarity': semantic_similarities[-1] if semantic_similarities else 0.0
            })
        
        # Calculate aggregate metrics
        total_pairs = len(test_qa_pairs)
        evaluation_results = {
            'total_qa_pairs': total_pairs,
            'successful_generations': success_count,
            'success_rate': round(success_count / total_pairs, 4),
            'exact_match_rate': round(exact_matches / total_pairs, 4),
            'partial_match_rate': round(partial_matches / total_pairs, 4),
            'average_response_time': round(np.mean(response_times), 4),
            'average_confidence': round(np.mean(confidence_scores), 4),
            'average_answer_length': round(np.mean(answer_lengths), 2),
            'semantic_similarity': round(np.mean(semantic_similarities), 4) if semantic_similarities else 0.0,
            'detailed_results': results
        }
        
        logger.info(f"Generation evaluation completed: Success rate: {evaluation_results['success_rate']}")
        return evaluation_results
    
    def _calculate_exact_match(self, expected: str, generated: str) -> bool:
        """Calculate exact match between expected and generated answers."""
        # Normalize text for comparison
        expected_norm = self._normalize_for_comparison(expected)
        generated_norm = self._normalize_for_comparison(generated)
        return expected_norm == generated_norm
    
    def _calculate_partial_match(self, expected: str, generated: str, threshold: float = 0.5) -> bool:
        """Calculate partial match based on word overlap."""
        expected_words = set(self._normalize_for_comparison(expected).split())
        generated_words = set(self._normalize_for_comparison(generated).split())
        
        if not expected_words:
            return False
        
        intersection = expected_words.intersection(generated_words)
        overlap_ratio = len(intersection) / len(expected_words)
        
        return overlap_ratio >= threshold
    
    def _calculate_semantic_similarity(self, expected: str, generated: str) -> float:
        """Calculate semantic similarity between texts."""
        if not SKLEARN_AVAILABLE:
            # Fallback to simple word overlap
            return self._simple_similarity(expected, generated)
        
        try:
            # Use TF-IDF for semantic similarity
            vectorizer = TfidfVectorizer()
            texts = [expected, generated]
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return self._simple_similarity(expected, generated)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity calculation."""
        words1 = set(self._normalize_for_comparison(text1).split())
        words2 = set(self._normalize_for_comparison(text2).split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.strip().lower())
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def evaluate_end_to_end(self, 
                           test_dataset: List[Dict[str, Any]],
                           save_results: bool = True,
                           output_path: str = "evaluation_results.json") -> EvaluationMetrics:
        """
        Perform comprehensive end-to-end evaluation.
        
        Args:
            test_dataset (List[Dict[str, Any]]): Test dataset with questions, answers, and relevant docs
            save_results (bool): Whether to save results to file
            output_path (str): Path to save results
            
        Returns:
            EvaluationMetrics: Comprehensive evaluation metrics
        """
        logger.info("Starting comprehensive end-to-end evaluation")
        start_time = time.time()
        
        # Separate retrieval and generation test data
        retrieval_queries = [
            {'question': item['question'], 'relevant_docs': item.get('relevant_docs', [])}
            for item in test_dataset if 'relevant_docs' in item
        ]
        
        qa_pairs = [
            {'question': item['question'], 'answer': item.get('answer', '')}
            for item in test_dataset if 'answer' in item
        ]
        
        # Evaluate retrieval
        retrieval_results = {}
        if retrieval_queries:
            retrieval_results = self.evaluate_retrieval(retrieval_queries)
        
        # Evaluate generation
        generation_results = {}
        if qa_pairs:
            generation_results = self.evaluate_generation(qa_pairs)
        
        # Create comprehensive metrics
        metrics = EvaluationMetrics(
            precision=retrieval_results.get('precision', 0.0),
            recall=retrieval_results.get('recall', 0.0),
            f1_score=retrieval_results.get('f1_score', 0.0),
            map_score=retrieval_results.get('map_score', 0.0),
            mrr_score=retrieval_results.get('mrr_score', 0.0),
            semantic_similarity=generation_results.get('semantic_similarity', 0.0),
            average_response_time=generation_results.get('average_response_time', 0.0),
            success_rate=generation_results.get('success_rate', 0.0),
            confidence_score=generation_results.get('average_confidence', 0.0),
            answer_accuracy=generation_results.get('exact_match_rate', 0.0),
            answer_completeness=generation_results.get('partial_match_rate', 0.0)
        )
        
        # Prepare comprehensive results
        evaluation_time = time.time() - start_time
        comprehensive_results = {
            'evaluation_timestamp': time.time(),
            'evaluation_duration_seconds': round(evaluation_time, 2),
            'test_dataset_size': len(test_dataset),
            'retrieval_evaluation': retrieval_results,
            'generation_evaluation': generation_results,
            'comprehensive_metrics': metrics.to_dict(),
            'system_info': self.rag_pipeline.get_system_stats()
        }
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(comprehensive_results, output_path)
        
        # Add to evaluation history
        self.evaluation_history.append(comprehensive_results)
        
        logger.info(f"End-to-end evaluation completed in {evaluation_time:.2f} seconds")
        return metrics
    
    def _save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def generate_evaluation_report(self, 
                                  metrics: EvaluationMetrics,
                                  output_path: str = "evaluation_report.txt") -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            metrics (EvaluationMetrics): Evaluation metrics
            output_path (str): Path to save the report
            
        Returns:
            str: Evaluation report text
        """
        report = f"""
# Arabic RAG System Evaluation Report

## System Overview
- Embedding Model: {self.rag_pipeline.embedding_model.model_name}
- Language Model: {self.rag_pipeline.llm_pipeline.generator.model_name}
- Chunking Strategy: {self.rag_pipeline.chunking_config.strategy}
- Vector Database: ChromaDB

## Retrieval Performance
- Precision: {metrics.precision:.4f}
- Recall: {metrics.recall:.4f}
- F1-Score: {metrics.f1_score:.4f}
- Mean Average Precision (MAP): {metrics.map_score:.4f}
- Mean Reciprocal Rank (MRR): {metrics.mrr_score:.4f}

## Generation Quality
- Answer Accuracy (Exact Match): {metrics.answer_accuracy:.4f}
- Answer Completeness (Partial Match): {metrics.answer_completeness:.4f}
- Semantic Similarity: {metrics.semantic_similarity:.4f}
- Average Confidence: {metrics.confidence_score:.4f}

## System Performance
- Success Rate: {metrics.success_rate:.4f}
- Average Response Time: {metrics.average_response_time:.4f} seconds

## Overall Assessment
"""
        
        # Add assessment based on metrics
        if metrics.f1_score >= 0.8:
            report += "- Retrieval Performance: Excellent\n"
        elif metrics.f1_score >= 0.6:
            report += "- Retrieval Performance: Good\n"
        elif metrics.f1_score >= 0.4:
            report += "- Retrieval Performance: Fair\n"
        else:
            report += "- Retrieval Performance: Needs Improvement\n"
        
        if metrics.answer_accuracy >= 0.8:
            report += "- Generation Quality: Excellent\n"
        elif metrics.answer_accuracy >= 0.6:
            report += "- Generation Quality: Good\n"
        elif metrics.answer_accuracy >= 0.4:
            report += "- Generation Quality: Fair\n"
        else:
            report += "- Generation Quality: Needs Improvement\n"
        
        if metrics.average_response_time <= 2.0:
            report += "- Response Time: Fast\n"
        elif metrics.average_response_time <= 5.0:
            report += "- Response Time: Acceptable\n"
        else:
            report += "- Response Time: Slow\n"
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
        
        return report
    
    def create_test_dataset(self, 
                           questions: List[str],
                           generate_answers: bool = True) -> List[Dict[str, Any]]:
        """
        Create a test dataset for evaluation.
        
        Args:
            questions (List[str]): List of test questions
            generate_answers (bool): Whether to generate reference answers
            
        Returns:
            List[Dict[str, Any]]: Test dataset
        """
        test_dataset = []
        
        for question in questions:
            test_item = {'question': question}
            
            if generate_answers:
                # Generate reference answer using the system
                result = self.rag_pipeline.query(question)
                test_item['answer'] = result.get('answer', '')
                test_item['confidence'] = result.get('confidence', 0.0)
            
            test_dataset.append(test_item)
        
        logger.info(f"Created test dataset with {len(test_dataset)} items")
        return test_dataset

if __name__ == "__main__":
    # Example usage would require a RAG pipeline instance
    print("Arabic RAG Evaluator module loaded successfully")
    print("Use with an initialized ArabicRAGPipeline instance for evaluation")