import logging
from typing import List, Dict, Optional, Union
import torch

# Try to import transformers
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GPT2LMHeadModel,
        GPT2Tokenizer,
        pipeline,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AraGPT2Generator:
    """
    AraGPT2 model integration for Arabic text generation in RAG system.
    Supports various AraGPT2 model variants and generation configurations.
    """
    
    def __init__(self, 
                 model_name: str = "aubmindlab/aragpt2-base",
                 device: Optional[str] = None,
                 max_length: int = 512,
                 cache_dir: Optional[str] = None):
        """
        Initialize AraGPT2 model.
        
        Args:
            model_name (str): Name of the AraGPT2 model variant
            device (str): Device to run the model on
            max_length (int): Maximum generation length
            cache_dir (str): Directory to cache the model
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required but not installed. Please install it with: pip install transformers")
        
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        self._load_model()
        logger.info(f"Initialized AraGPT2Generator with {model_name} on {self.device}")
    
    def _get_best_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the AraGPT2 model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Set up generation configuration
            self.generation_config = GenerationConfig(
                max_length=self.max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                length_penalty=1.0
            )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_text(self, 
                     prompt: str,
                     max_new_tokens: int = 200,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     repetition_penalty: float = 1.2,
                     do_sample: bool = True) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt (str): Input prompt for generation
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            top_k (int): Top-k sampling parameter
            repetition_penalty (float): Repetition penalty
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated text
        """
        if not prompt.strip():
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - max_new_tokens
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return ""
    
    def generate_answer(self, 
                       question: str,
                       context: str,
                       max_new_tokens: int = 150,
                       temperature: float = 0.5) -> str:
        """
        Generate an answer based on question and context (RAG).
        
        Args:
            question (str): Question to answer
            context (str): Context information
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Generation temperature
            
        Returns:
            str: Generated answer
        """
        # Create RAG prompt
        prompt = self._create_rag_prompt(question, context)
        
        # Generate answer
        answer = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        return answer.strip()
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create a RAG prompt combining question and context.
        
        Args:
            question (str): Question to answer
            context (str): Context information
            
        Returns:
            str: Formatted RAG prompt
        """
        # Arabic RAG prompt template
        prompt_template = """السياق: {context}

السؤال: {question}

الإجابة:"""
        
        # Format the prompt
        prompt = prompt_template.format(
            context=context.strip(),
            question=question.strip()
        )
        
        return prompt
    
    def generate_multiple_answers(self, 
                                 question: str,
                                 contexts: List[str],
                                 max_new_tokens: int = 150,
                                 temperature: float = 0.5) -> List[Dict[str, str]]:
        """
        Generate multiple answers using different contexts.
        
        Args:
            question (str): Question to answer
            contexts (List[str]): List of context information
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Generation temperature
            
        Returns:
            List[Dict[str, str]]: List of answers with contexts
        """
        answers = []
        
        for i, context in enumerate(contexts):
            try:
                answer = self.generate_answer(
                    question=question,
                    context=context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                answers.append({
                    'context_id': i,
                    'context': context,
                    'answer': answer,
                    'question': question
                })
                
            except Exception as e:
                logger.error(f"Failed to generate answer for context {i}: {e}")
                answers.append({
                    'context_id': i,
                    'context': context,
                    'answer': "",
                    'question': question,
                    'error': str(e)
                })
        
        return answers
    
    def generate_summary(self, 
                        text: str,
                        max_new_tokens: int = 100,
                        temperature: float = 0.3) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text (str): Text to summarize
            max_new_tokens (int): Maximum new tokens for summary
            temperature (float): Generation temperature
            
        Returns:
            str: Generated summary
        """
        # Arabic summarization prompt
        prompt = f"""النص التالي: {text}

ملخص النص:"""
        
        summary = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        return summary.strip()
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]],
                       max_new_tokens: int = 200,
                       temperature: float = 0.7) -> str:
        """
        Generate response for chat-like interaction.
        
        Args:
            messages (List[Dict[str, str]]): List of messages with 'role' and 'content'
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Generation temperature
            
        Returns:
            str: Generated response
        """
        # Convert messages to prompt format
        prompt = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                prompt += f"المستخدم: {content}\n"
            elif role == 'assistant':
                prompt += f"المساعد: {content}\n"
            elif role == 'system':
                prompt += f"النظام: {content}\n"
        
        prompt += "المساعد:"
        
        response = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Union[str, int]]: Model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'model_size': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def save_model(self, save_path: str):
        """
        Save the model and tokenizer.
        
        Args:
            save_path (str): Path to save the model
        """
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

class AraGPT2Pipeline:
    """
    High-level pipeline for AraGPT2 operations.
    """
    
    def __init__(self, model_name: str = "aubmindlab/aragpt2-base"):
        """
        Initialize the pipeline.
        
        Args:
            model_name (str): Name of the AraGPT2 model
        """
        self.generator = AraGPT2Generator(model_name=model_name)
        logger.info("AraGPT2Pipeline initialized")
    
    def answer_question(self, 
                       question: str, 
                       retrieved_docs: List[Dict[str, str]],
                       max_contexts: int = 3) -> Dict[str, Union[str, List]]:
        """
        Answer a question using retrieved documents.
        
        Args:
            question (str): Question to answer
            retrieved_docs (List[Dict[str, str]]): Retrieved documents
            max_contexts (int): Maximum number of contexts to use
            
        Returns:
            Dict[str, Union[str, List]]: Answer and metadata
        """
        if not retrieved_docs:
            return {
                'answer': 'لم أجد معلومات كافية للإجابة على هذا السؤال.',
                'contexts_used': [],
                'confidence': 0.0
            }
        
        # Select top contexts
        top_contexts = retrieved_docs[:max_contexts]
        contexts = [doc.get('text', '') for doc in top_contexts]
        
        # Combine contexts
        combined_context = ' '.join(contexts)
        
        # Generate answer
        answer = self.generator.generate_answer(
            question=question,
            context=combined_context,
            max_new_tokens=150,
            temperature=0.5
        )
        
        return {
            'answer': answer,
            'contexts_used': top_contexts,
            'confidence': self._calculate_confidence(answer, contexts),
            'question': question
        }
    
    def _calculate_confidence(self, answer: str, contexts: List[str]) -> float:
        """
        Calculate confidence score for the answer.
        
        Args:
            answer (str): Generated answer
            contexts (List[str]): Contexts used
            
        Returns:
            float: Confidence score (0-1)
        """
        if not answer or not contexts:
            return 0.0
        
        # Simple confidence calculation based on answer length and context overlap
        answer_length = len(answer.split())
        context_length = sum(len(ctx.split()) for ctx in contexts)
        
        # Basic heuristic: longer answers with more context tend to be more confident
        base_confidence = min(answer_length / 50, 1.0)  # Normalize by expected answer length
        context_boost = min(context_length / 500, 0.3)  # Boost for having more context
        
        confidence = min(base_confidence + context_boost, 1.0)
        return round(confidence, 2)

if __name__ == "__main__":
    # Example usage
    try:
        generator = AraGPT2Generator()
        
        # Test text generation
        prompt = "الذكاء الاصطناعي هو"
        generated = generator.generate_text(prompt, max_new_tokens=50)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        
        # Test RAG answer generation
        question = "ما هو الذكاء الاصطناعي؟"
        context = "الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات المبرمجة للتفكير والتعلم مثل البشر."
        answer = generator.generate_answer(question, context)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        # Model info
        info = generator.get_model_info()
        print(f"Model info: {info}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install the required dependencies and have sufficient memory/GPU resources.")