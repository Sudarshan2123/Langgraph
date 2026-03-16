
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tools import tools
from src.pipeline.intent_classifier import IntentClassifier, GreetingGenerator, OutOfScopeHandler, UnclearHandler
from src.config.configuration import ConfigurationManager
from langchain_google_genai import ChatGoogleGenerativeAI
from config.Authentication.gcp import load_gcp_credentials

class Init:
    def __init__(self):
        self.config_obj = ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        credentials = load_gcp_credentials()

        self.vertex_llm = ChatGoogleGenerativeAI(
            model=self.config.RAG_MODEL,
            temperature=0.0,
            max_output_tokens=4096,
            credentials=credentials,
            max_retries=2
        )

        self.intent_llm = ChatNVIDIA(model="qwen/qwen3.5-122b-a10b",temperature=0.2,nvidia_api_key="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")
        self.intent_classifier = IntentClassifier(self.intent_llm)
        self.intent_classifier.clear_cache()
        self.greeting_generator = GreetingGenerator(self.intent_llm)
        self.out_of_scope_handler = OutOfScopeHandler(self.intent_llm)
        self.unclear_handler = UnclearHandler(self.intent_llm)
  
        # self.Analyzer = DataAnalyzer(config=self.config, vertex_llm=self.vertex_llm,engine=self.engine)


# ✅ Initialize ONCE at module load time — this is the key
_pipeline = Init()

def get_pipeline() -> Init:
    return _pipeline  # Always return the same object, no Init() call