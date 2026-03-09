import asyncio
import os
import logging
from contextlib import asynccontextmanager
from langchain_core.messages.ai import AIMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from agentState import AgentState
from singleton import get_pipeline
from tools import tools as local_tools
from mcp_loader import init_mcp_session, close_mcp_session, get_mcp_tools
import translate

SIMPLE_GREETINGS = [
    'hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you'
]

LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "Langgraph"


# --- Helper Functions ---
def translate_input(user_input: str, lang: str) -> str:
    if lang != "en-US":
        if user_input.lower().strip() in SIMPLE_GREETINGS:
            return user_input
        return translate.Translate_process_chat(user_input, "en")
    return user_input


# --- LangGraph Node Functions ---
def translate_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    if isinstance(last_message, dict):
        user_input = last_message.get("content", "")
        lang = last_message.get("lang", "en-US")
    else:
        user_input = last_message.content
        lang = getattr(last_message, "lang", "en-US")
    translated = translate_input(user_input, lang)
    return {"messages": state["messages"][:-1] + [HumanMessage(content=translated)]}


async def classifier_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    logging.info(f"Last message: {last_message}")
    pipeline = get_pipeline()
    # ✅ asyncio.to_thread instead of get_event_loop().run_in_executor
    classification = await asyncio.to_thread(
        pipeline.intent_classifier.classify_intent,
        last_message
    )
    return {"classification": classification}


def route_after_classification(state: AgentState):
    classification = state["classification"]
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

def get_all_tools():
    return local_tools + get_mcp_tools()

def make_tool_node():
    """Called after lifespan so MCP tools are already loaded."""
    return ToolNode(get_all_tools())

def assistant_node(state: AgentState) -> dict:
    pipeline = get_pipeline()
    classification = state["classification"]

    extracted_query = (
        classification.extracted_query
        or state["messages"][-1].content
    )

    # ✅ get_mcp_tools() is now sync — tools already loaded by lifespan
    all_tools = get_all_tools()
    llm = pipeline.intent_llm.bind_tools(all_tools)

    sys_msg = SystemMessage(content=(
        "You are a helpful assistant with access to tools. "
        "Analyze the user's query and call the most appropriate tool. "
        "Pass the user's question as the 'user_input' argument to the tool."
    ))

    return {"messages": [llm.invoke([sys_msg, HumanMessage(content=extracted_query)])]}


# --- Lifespan: init MCP inside LangGraph's event loop ---
@asynccontextmanager
async def app_lifespan(app):
    await init_mcp_session() 
    builder.add_node("tools", make_tool_node()) # ✅ MCP session created in correct loop
    yield
    await close_mcp_session()  # ✅ clean shutdown


# --- Build Graph ---
builder = StateGraph(AgentState)

builder.add_node("translate_input", translate_node)
builder.add_node("intent_classifier", classifier_node)
builder.add_node("assistant", assistant_node)
builder.add_node("handle_greeting", greeting_handler_node)
builder.add_node("handle_out_of_scope", out_of_scope_handler_node)
builder.add_node("handle_unclear", unclear_handler_node)
# ✅ Placeholder — will be replaced in lifespan with real tools
builder.add_node("tools", ToolNode(local_tools))

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