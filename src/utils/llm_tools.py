from utils.config import *
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

class LLMTools():
    def initialize_llm(self, provider_choice=1, model_name="google/gemini-2.5-flash"):
        
        # Configuration constants
        OPENROUTER_API_KEY = OPEN_ROUTER_Apexchat_API_KEY
        OPENROUTER_MODEL = model_name
        
        OLLAMA_BASE_URL = OLLAMA_URL
        OLLAMA_MODEL = model_name
        
        if provider_choice == 1:
            # OpenRouter LLM
            llm = ChatOpenAI(
                model=OPENROUTER_MODEL,
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            print(f"Initialized OpenRouter LLM with model: {OPENROUTER_MODEL}")
            
        elif provider_choice == 2:
            # Ollama LLM
            llm = OllamaLLM(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
            )
            print(f"Initialized Ollama LLM with model: {OLLAMA_MODEL}")
            
        else:
            raise ValueError("provider_choice must be 1 (OpenRouter) or 2 (Ollama)")
        
        return llm