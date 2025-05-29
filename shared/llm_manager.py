import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    _instance = None
    _llama_pipeline = None
    _llama_index_llm = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.model_name = "meta-llama/Llama-2-7b-chat-hf"
            self.initialized = False
            
    def load_models(self):
        """Load both pipeline and LlamaIndex LLM models"""
        if self.initialized:
            return
            
        try:
            logger.info("Loading Llama-2 models...")
            
            # Load tokenizer and model for pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad_token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                load_in_8bit=True
            )
            
            # Create pipeline for API agent
            self._llama_pipeline = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            # Create custom LlamaIndex-style LLM wrapper
            self._llama_index_llm = self._create_llama_index_wrapper()
            
            self.initialized = True
            logger.info("✅ LLM models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _create_llama_index_wrapper(self):
        """Create a wrapper that mimics LlamaIndex HuggingFaceLLM interface"""
        class LlamaIndexLLMWrapper:
            def __init__(self, pipeline, system_prompt):
                self.pipeline = pipeline
                self.system_prompt = system_prompt
                self.context_window = 4096
                self.max_new_tokens = 256
            
            def complete(self, prompt, **kwargs):
                """Generate completion for a prompt"""
                try:
                    # Combine system prompt with user prompt
                    full_prompt = f"{self.system_prompt}\n\n<|USER|>{prompt}<|ASSISTANT|>"
                    
                    # Generate response
                    outputs = self.pipeline(
                        full_prompt,
                        max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
                        temperature=kwargs.get('temperature', 0.0),
                        do_sample=kwargs.get('do_sample', False),
                        pad_token_id=self.pipeline.tokenizer.eos_token_id,
                        return_full_text=False,
                        num_return_sequences=1
                    )
                    
                    generated_text = outputs[0]['generated_text']
                    return generated_text.strip()
                    
                except Exception as e:
                    logger.error(f"Error in LLM completion: {e}")
                    return f"Error generating response: {str(e)}"
            
            def chat(self, messages, **kwargs):
                """Handle chat-style interactions"""
                # Convert messages to prompt format
                prompt_parts = []
                for message in messages:
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    
                    if role == 'system':
                        prompt_parts.append(f"System: {content}")
                    elif role == 'user':
                        prompt_parts.append(f"Human: {content}")
                    elif role == 'assistant':
                        prompt_parts.append(f"Assistant: {content}")
                
                prompt = "\n".join(prompt_parts)
                return self.complete(prompt, **kwargs)
            
            def __call__(self, prompt, **kwargs):
                """Make the object callable"""
                return self.complete(prompt, **kwargs)
        
        # Define system prompt
        system_prompt = """
        You are a financial data extraction assistant. Your task is to accurately extract key financial metrics from company reports and related documents.
        Focus specifically on:
        - company name 
        - Total Revenue or Net Sales
        - Revenue growth (Year-over-Year or Quarter-over-Quarter, if available)
        - Operating Profit or EBITDA
        - Net Profit
        - Segment-wise revenue breakdowns (if present)
        - Currency of the reported values
        If any information is missing or not found in the context, clearly state: "Not available in the document."
        Be concise and use bullet points where applicable.
        """
        
        return LlamaIndexLLMWrapper(self._llama_pipeline, system_prompt)
    
    @property
    def pipeline(self):
        """Get the text generation pipeline"""
        if not self.initialized:
            self.load_models()
        return self._llama_pipeline
    
    @property 
    def llama_index_llm(self):
        """Get the LlamaIndex-style LLM wrapper"""
        if not self.initialized:
            self.load_models()
        return self._llama_index_llm
    
    def generate_response(self, prompt, max_length=512, temperature=0.7):
        """Convenience method for generating responses"""
        if not self.initialized:
            self.load_models()
            
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            return outputs[0]['generated_text'].strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

# Global instance
llm_manager = LLMManager()

# Helper classes for backward compatibility (if needed elsewhere in your code)
class ChatMessage:
    def __init__(self, role=None, content=None, **kwargs):
        self.role = role
        self.content = content
        for k, v in kwargs.items():
            setattr(self, k, v)

class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"