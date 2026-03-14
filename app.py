import asyncio
import os
import logging
from langchain_core.messages.ai import AIMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from agentState import AgentState
from singleton import get_pipeline
from tools import tools as local_tools
from mcp_loader import init_mcp_session, get_mcp_tools
import translate

SIMPLE_GREETINGS = [
    'hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you'
]

LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "Langgraph"


def get_all_tools():
    return get_mcp_tools() + local_tools 


def translate_input(user_input: str, lang: str) -> str:
    if lang != "en-US":
        if user_input.lower().strip() in SIMPLE_GREETINGS:
            return user_input
        return translate.Translate_process_chat(user_input, "en")
    return user_input


def translate_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    if isinstance(last_message, dict):
        user_input = last_message.get("content", "")
        lang = last_message.get("lang", "en-US")
    else:
        user_input = last_message.content
        lang = getattr(last_message, "lang", "en-US")

    if lang == "en-US" or user_input.lower().strip() in SIMPLE_GREETINGS:
        return {}

    translated = translate_input(user_input, lang)
    return {"messages": [RemoveMessage(id=last_message.id), HumanMessage(content=translated)]}


async def classifier_node(state: AgentState) -> dict:
    # ✅ Init MCP here — runs inside LangGraph's event loop, idempotent after first call
    await init_mcp_session()

    raw = state["messages"][-1]
    last_message = raw.get("content", "") if isinstance(raw, dict) else raw.content  # ← fix here

    pipeline = get_pipeline()
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

    
def assistant_node(state: AgentState) -> dict:
    pipeline = get_pipeline()
    classification = state["classification"]

    # raw = state["messages"][-1]
    # raw_query = raw.get("content", "") if isinstance(raw, dict) else raw.content  # ← fix here

    # extracted_query = str(raw_query.get("value", raw_query) if isinstance(raw_query, dict) else raw_query)

    all_tools = get_all_tools()
    logging.info(all_tools)
    llm = pipeline.vertex_llm.bind_tools(all_tools)

    system_content = (
        "You are an intelligent Assitant. Use the provided tools to answer the user and invoke it.\n"
        f"INTENT CONTEXT: The classifier identifies this as '{classification.primary_intent}'.\n"
        "Instructions: Prioritize the suggested tool. Ensure 'user_input' is a simple string.\n"
        "follow the input scheme provided by the tool strictly.\n"
        "Check the provided tool result and presented it to the user directly is required.\n"
        "You are a intelligent assistant if the user query require calling tool multiple times to satisfy the user query call it like if user ask dress code and leave policy then you require to call tool to answer both dress code and leave policy.\n"
        "Once You are satisfied with the process then combine the answer of the tools used/required and provided it as plain text answer "
    )

    messages = [SystemMessage(content=system_content)]
    messages.extend(state["messages"][-5:])#adjust the history box number

    # if not (state["messages"] and isinstance(state["messages"][-1], AIMessage)):
    #     messages.append(HumanMessage(content=extracted_query))

    response = llm.invoke(messages)
    return {"messages": [response]}



class DynamicToolNode:
    """Wraps ToolNode but rebuilds with current tools on each call."""
    async def __call__(self, state):
        node = ToolNode(get_all_tools())
        return await node.ainvoke(state)


# --- Build Graph ---
builder = StateGraph(AgentState)

builder.add_node("translate_input", translate_node)
builder.add_node("intent_classifier", classifier_node)
builder.add_node("assistant", assistant_node)
builder.add_node("handle_greeting", greeting_handler_node)
builder.add_node("handle_out_of_scope", out_of_scope_handler_node)
builder.add_node("handle_unclear", unclear_handler_node)
builder.add_node("tools", DynamicToolNode()) 

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