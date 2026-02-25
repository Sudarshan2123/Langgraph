# from logging import Logger

from langchain_ollama import ChatOllama
from tools import tools
from src.pipeline.intent_classifier import IntentClassifier,GreetingGenerator,OutOfScopeHandler
from src.config.configuration import ConfigurationManager

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
    # credentials = get_gcp_credentials()
    # self.llm = ChatVertexAI(
    #     model_name=self.config.RAG_MODEL,
    #     temperature=0.4,  # Lower temp for consistent classification
    #     max_output_tokens=2500,
    #     credentials=credentials
    # )
    # self.rag_instance = RAGPipeline()
    self.llm = ChatOllama(model="qwen3:1.7b").bind_tools(tools)
    self.intent_classifier = IntentClassifier(self.llm)
    self.intent_classifier.clear_cache()
    self.greeting_generator = GreetingGenerator(self.llm)
    self.out_of_scope_handler = OutOfScopeHandler(self.llm)
    # self.conversation_manager = ConversationManager(self.config)
    # self.session_manager = SessionManager(self.config)
    # self.streaming_chatbot = StreamingChatbot(self.intent_llm)

    # Logger.info("ChatbotPipeline initialized with dynamic intent " "classification and streaming")

def get_pipeline() -> Init:
  return Init()

    