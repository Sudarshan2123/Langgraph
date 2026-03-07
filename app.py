import asyncio
import os
import logging
from langchain_core.messages.ai import AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from agentState import AgentState
from pipeline.intent_classifier import IntentClassification
from singleton import get_pipeline
from tools import tools as local_tools
from mcp_loader import get_mcp_tools
import translate

SIMPLE_GREETINGS = [
    'hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you'
]

# LangSmith configuration
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "Langgraph"


# --- Helper Functions ---
def translate_input(user_input: str, lang: str) -> str:
    """Translate input text if needed."""
    if lang != "en-US":
        if user_input.lower().strip() in SIMPLE_GREETINGS:
            return user_input
        return translate.Translate_process_chat(user_input, "en")
    return user_input


# --- LangGraph Node Functions ---
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
    updated_message = HumanMessage(content=translated)
    return {"messages": state["messages"][:-1] + [updated_message]}


async def classifier_node(state: AgentState) -> dict:
    """Intent classification node."""
    last_message = state["messages"][-1].content
    logging.info(f"Last message: {last_message}")
    pipeline = get_pipeline()
    classification = await asyncio.get_event_loop().run_in_executor(
        None,
        pipeline.intent_classifier.classify_intent,
        last_message
    )
    return {"classification": classification}


def route_after_classification(state: AgentState):
    classification = state["classification"]

    # If tool call required, always route to assistant regardless of intent
    if hasattr(classification, "requires_tool_call") and classification.requires_tool_call:
        return "assistant"

    if classification.primary_intent == "greeting":
        return "handle_greeting"

    if classification.primary_intent == "out_of_scope":
        return "handle_out_of_scope"

    if classification.primary_intent == "unclear":
        return "handle_unclear"

    return "assistant"


def greeting_handler_node(state: AgentState) -> dict:
    pipeline = get_pipeline()
    last_msg = state["messages"][-1].content
    response = pipeline.greeting_generator.generate_greeting(
        last_msg,
        state["classification"].greeting_type or "casual"
    )
    return {"messages": [AIMessage(content=response)]}


def out_of_scope_handler_node(state: AgentState) -> dict:
    pipeline = get_pipeline()
    last_msg = state["messages"][-1].content
    response = pipeline.out_of_scope_handler.handle_out_of_scope(last_msg)
    return {"messages": [AIMessage(content=response)]}


def unclear_handler_node(state: AgentState) -> dict:
    pipeline = get_pipeline()
    last_msg = state["messages"][-1].content
    response = pipeline.unclear_handler.handle_unclear(last_msg)
    return {"messages": [AIMessage(content=response)]}


def finalResponder(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        return {"messages": [AIMessage(content=last_message.content)]}


# --- Load MCP Tools Synchronously ---
def _load_tools_sync():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(get_mcp_tools())
    except Exception as e:
        logging.warning(f"MCP tools load failed: {e}")
        return []


mcp_tools = _load_tools_sync()
all_tools = local_tools + mcp_tools

if mcp_tools:
    logging.info(f"Loaded MCP tools: {[t.name for t in mcp_tools]}")
else:
    logging.warning("No MCP tools loaded — Policy_RAG_Implementation unavailable")



def assistant_node(state: AgentState) -> dict:
    """Assistant node — routes to tools based on classification."""
    pipeline = get_pipeline()
    classification = state["classification"]

    extracted_query = (
        classification.extracted_query
        or state["messages"][-1].content
    )

    # Bind all tools fresh from pipeline
    llm = pipeline.llm.bind_tools(all_tools)  # use pipeline llm with all tools

    sys_msg = SystemMessage(content=(
        "You are an Assistant that invokes tools with the exact user query. "
        "When calling Policy_RAG_Implementation, pass the user's question as user_input. "
        "When calling structure_data, pass the user's question as user_input. "
        f"The user's query is: {extracted_query}. "
        "Call the appropriate tool now with this exact query string as user_input."
    ))
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


# --- Build Graph ---
builder = StateGraph(AgentState)

builder.add_node("translate_input", translate_node)
builder.add_node("intent_classifier", classifier_node)
builder.add_node("assistant", assistant_node)
builder.add_node("handle_greeting", greeting_handler_node)
builder.add_node("handle_out_of_scope", out_of_scope_handler_node)
builder.add_node("handle_unclear", unclear_handler_node)
builder.add_node("tools", ToolNode(all_tools))

builder.add_edge(START, "translate_input")
builder.add_edge("translate_input", "intent_classifier")
builder.add_conditional_edges("intent_classifier", route_after_classification, {
    "handle_greeting": "handle_greeting",
    "handle_out_of_scope": "handle_out_of_scope",
    "handle_unclear": "handle_unclear",
    "assistant": "assistant"
})
builder.add_edge("handle_greeting", END)
builder.add_edge("handle_out_of_scope", END)
builder.add_edge("handle_unclear", END)
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()