# from logging import Logger

import os
from typing import Optional
from langchain_ollama import ChatOllama
from sqlalchemy import Engine
from pipeline.Data_Analyzer import DataAnalyzer
from tools import tools
from src.pipeline.intent_classifier import IntentClassifier,GreetingGenerator,OutOfScopeHandler,UnclearHandler
from pipeline.data_extract import DataBase
from src.config.configuration import ConfigurationManager
from langchain_google_genai import ChatGoogleGenerativeAI
from config.Authentication.gcp import load_gcp_credentials



class Init :
  _instance= None

  def __new__(cls,*args,**kwargs):
    if cls._instance is None:
        cls._instance = super().__new__(cls)
    return cls._instance

  def __init__(self):
    """Initialize the chatbot pipeline with all necessary components."""
    if hasattr(self,'_initialized'):
        return

    self.config_obj = ConfigurationManager()
    self.config = self.config_obj.get_base_config()
    credentials = load_gcp_credentials()
    self.POSTGRES_USER = os.environ.get("POSTGRES_USER", self.config.POSTGRES_USER)
    self.POSTGRES_HOST = os.environ.get("POSTGRES_HOST", self.config.POSTGRES_HOST)
    self.POSTGRES_PORT = os.environ.get("POSTGRES_PORT", self.config.POSTGRES_PORT)
    self.POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", self.config.POSTGRES_PASSWORD)
    self.POSTGRES_DB = os.environ.get("POSTGRES_DB", self.config.POSTGRES_DB)
        
        # Redis Configuration
    self.REDIS_USERNAME = os.environ.get("REDIS_USERNAME", getattr(self.config, 'REDIS_USERNAME', 'redis'))
    self.REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", getattr(self.config, 'REDIS_PASSWORD', 'None'))
    self.REDIS_HOST = os.environ.get("REDIS_HOST", getattr(self.config, 'REDIS_HOST', 'localhost'))
    self.REDIS_PORT = int(os.environ.get("REDIS_PORT", getattr(self.config, 'REDIS_PORT', 6379)))
    self.REDIS_DB = int(os.environ.get("REDIS_DB", getattr(self.config, 'REDIS_DB', 0)))
    self.CACHE_TTL = os.environ.get("CACHE_TTL", getattr(self.config, 'CACHE_TTL', None))
    self.vertex_llm = ChatGoogleGenerativeAI(
                model_name=self.config.RAG_MODEL,
                temperature=0.0,
                max_output_tokens=4096,  # Increased back for complete responses
                credentials=credentials,
                max_retries=2
            )
    # Uncomment above and remove/comment ChatOllama below to use Gemini
    # self.rag_instance = RAGPipeline()
    self.llm = ChatOllama(model="deepseek-r1:8b",temperature=0.2).bind_tools(tools)
    self.intent_llm = ChatOllama(model="llama3.2:latest", temperature=0.1) 
    self.intent_classifier = IntentClassifier(self.intent_llm)
    self.intent_classifier.clear_cache()
    self.greeting_generator = GreetingGenerator(self.intent_llm)
    self.out_of_scope_handler = OutOfScopeHandler(self.intent_llm)
    self.unclear_handler = UnclearHandler(self.intent_llm) 
    self.engine = DataBase.connect()
    self.Database = DataBase(self)
    self.Analyzer = DataAnalyzer(config=self.config,vertex_llm=self.vertex_llm)
    # self.conversation_manager = ConversationManager(self.config)
    # self.session_manager = SessionManager(self.config)
    # self.streaming_chatbot = StreamingChatbot(self.intent_llm)

    # Logger.info("ChatbotPipeline initialized with dynamic intent " "classification and streaming")

def get_pipeline() -> Init:
  return Init()

    