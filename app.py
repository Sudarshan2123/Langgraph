import os
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState,END
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from classifier import IntentClassification
from singleton import get_pipeline
from tools import tools
import translate


SIMPLE_GREETINGS = [
    'hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you'
]

# LangSmith configuration
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "Langgraph"


def translate_input(user_input: str, lang: str) -> str:
    """Translate input text if needed."""
    if lang != "en-US":
        if user_input.lower().strip() in SIMPLE_GREETINGS:
            return user_input
        return translate.Translate_process_chat(user_input, "en")
    return user_input  # Fixed: was missing return for English input


def intent_classifier(translated_input: str) -> IntentClassification:
    """
    Classify user intent with retry logic.

    Args:
        translated_input: User's translated message

    Returns:
        IntentClassification object
    """
    pipeline = get_pipeline()
    classification = pipeline.intent_classifier.classify_intent(translated_input)
    return classification  # Fixed: was missing return


# --- LangGraph Node Functions ---
# Node functions must accept and return state dicts

def translate_node(state: MessagesState) -> dict:
    """Translation node: translates the latest user message."""
    last_message = state["messages"][-1]
    user_input = last_message.content

    # Detect language from message metadata if available, default to "en-US"
    lang = getattr(last_message, "lang", "en-US")

    translated = translate_input(user_input, lang)

    # Replace the last message content with translated text
    updated_message = HumanMessage(content=translated)
    return {"messages": state["messages"][:-1] + [updated_message]}


def classifier_node(state: MessagesState) -> dict:
    """Intent classification node."""
    last_message = state["messages"][-1]
    classification = intent_classifier(last_message.content)

    # Return classification result as an AI message
    response_text = f"Intent classified as: {classification.primary_intent}"
    return {"messages": state["messages"] + [AIMessage(content=response_text)]}

def assistant_node(state:MessagesState) -> dict:
    """ Classified intent is pass to assitant to redirect it to tools"""
    sys_msg = SystemMessage(content=("You are an Assistant whose job is to invoke the required tools "
                                     "and return the output in a proper format."
    ))
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


llm = ChatOllama(model="llama3.2:latest").bind_tools(tools)
# --- Build Graph ---
builder = StateGraph(MessagesState)

# Fixed: node names must match what's used in add_edge/add_conditional_edges
builder.add_node("translate_input", translate_node)
builder.add_node("intent_classifier", classifier_node)
builder.add_node("assistant",assistant_node)
builder.add_node("tools", ToolNode(tools))

# Fixed: added START edge and corrected node name references
builder.add_edge(START, "translate_input")
builder.add_edge("translate_input", "intent_classifier")
builder.add_edge("intent_classifier", "assistant")
builder.add_conditional_edges("assistant", tools_condition) 
builder.add_edge("tools","assistant")
builder.add_edge("assistant",END) # Loop back after tool execution

# Compile graph
graph = builder.compile()