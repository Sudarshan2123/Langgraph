from os import environ
from langgraph.graph import StateGraph,START,MessagesState
from langchain_core.messages import SystemMessage, tool
from langgraph.prebuilt import tool_node,tools_condition
import translate
from classifier import IntentClassification

SIMPLE_GREETINGS = [
    'hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you'
]

def translate_input(user_input:str,lang:str) ->str:
    """translate input test if needed"""
    if lang!="en-US":
        if user_input.lower().strip() in SIMPLE_GREETINGS:
            return user_input
    return translate.translate_process_chat(user_input,"en")

def intent_classifier(user_input:str) ->IntentClassification:
    """
    Classify user intent with retry logic.

    Args:
      user_input:User's message

    Returns:
      IntentClassification object
    """
    intent_key = user_input.lower().strip()

    if intent_key in self._classification_cache:
        return self._classification_cache[intent_key]
    
    try:
        result = self