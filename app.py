import asyncio
import os
from langchain_core.messages.ai import AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START, MessagesState
from langchain_core.messages import HumanMessage,SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from agentState import AgentState
from pipeline.intent_classifier import IntentClassification
from singleton import get_pipeline
from tools import tools
import translate
import logging

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
    logging.info(f"Translated input: {translated_input}")
    classification = pipeline.intent_classifier.classify_intent(translated_input)
    return classification  # Fixed: was missing return


# --- LangGraph Node Functions ---
# Node functions must accept and return state dicts

def translate_node(state: AgentState) -> dict:
    """Translation node: translates the latest user message."""
    last_message = state["messages"][-1]
    if isinstance(last_message, dict):
        user_input = last_message.get("content", "")
        lang = last_message.get("lang", "en-US")
    else:
        user_input = last_message.content
        lang = getattr(last_message, "lang", "en-US")
    translated = translate_input(user_input, lang)

    # Replace the last message content with translated text
    updated_message = HumanMessage(content=translated)
    return {"messages": state["messages"][:-1] + [updated_message]}


async def classifier_node(state: AgentState) -> dict:
    """Intent classification node."""
    last_message = state["messages"][-1].content
    logging.info(f"Last message: {last_message}")
    
    pipeline = get_pipeline()
    # Run blocking LLM call in executor so it doesn't block the event loop
    classification = await asyncio.get_event_loop().run_in_executor(
        None, 
        pipeline.intent_classifier.classify_intent, 
        last_message
    )
    return {"classification": classification}

def route_after_classification(state: AgentState):
    classification = state["classification"]
    
    if classification.primary_intent == "greeting" or classification.secondary_intent == "greeting":
        return "handle_greeting"
    
    if classification.primary_intent == "out_of_scope":
        return "handle_out_of_scope"
    
    if classification.primary_intent == "unclear":
        return "handle_unclear"
    
    return "assistant"

def greeting_handler_node(state: AgentState):
    pipeline = get_pipeline()
    last_msg = state["messages"][-1].content
    response = pipeline.greeting_generator.generate_greeting(
        last_msg, 
        state["classification"].greeting_type or "casual"
    )
    return {"messages": [AIMessage(content=response)]}

def out_of_scope_handler_node(state: AgentState):
    pipeline = get_pipeline()
    last_msg = state["messages"][-1].content
    response = pipeline.out_of_scope_handler.handle_out_of_scope(last_msg)
    return {"messages": [AIMessage(content=response)]}

def unclear_handler_node(state: AgentState):
    pipeline = get_pipeline()
    last_msg = state["messages"][-1].content
    response = pipeline.unclear_handler.handle_unclear(last_msg)
    return {"messages": [AIMessage(content=response)]}

def assistant_node(state:AgentState) -> dict:
    """ Classified intent is pass to assitant to redirect it to tools"""
    sys_msg = SystemMessage(content=("You are an Assistant whose job is to invoke the required tools "
                                     "and return the response output in a exact same format as give by the tools."
    ))
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

def finalResponder(state:AgentState) -> dict:
    last_message = state["messages"][-1]

    if isinstance(last_message,ToolMessage):
        return {"messages":[AIMessage(content=last_message.content)]}


llm = ChatOllama(model="llama3.2:latest",temperature=0.1).bind_tools(tools)
# --- Build Graph ---
builder = StateGraph(AgentState)

# Fixed: node names must match what's used in add_edge/add_conditional_edges
builder.add_node("translate_input", translate_node)
builder.add_node("intent_classifier", classifier_node)
builder.add_node("assistant",assistant_node)
builder.add_node("handle_greeting",greeting_handler_node)
builder.add_node("handle_out_of_scope",out_of_scope_handler_node)
builder.add_node("handle_unclear", unclear_handler_node)
builder.add_node("tools", ToolNode(tools))

# Fixed: added START edge and corrected node name references
builder.add_edge(START, "translate_input")
builder.add_edge("translate_input", "intent_classifier")
builder.add_conditional_edges("intent_classifier", route_after_classification, {
    "handle_greeting":"handle_greeting",
    "handle_out_of_scope":"handle_out_of_scope",
    "handle_unclear": "handle_unclear",
    "assistant":"assistant"
}) 
builder.add_edge("handle_greeting",END)
builder.add_edge("handle_out_of_scope",END)
builder.add_edge("handle_unclear", END)
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()