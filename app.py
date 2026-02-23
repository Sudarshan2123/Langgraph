from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

import translate
from classifier import IntentClassification
from singleton import get_pipeline


SIMPLE_GREETINGS = [
    'hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you'
]


def translate_input(user_input: str, lang: str) -> str:
    """Translate input text if needed."""
    if lang != "en-US":
        if user_input.lower().strip() in SIMPLE_GREETINGS:
            return user_input
        return translate.translate_process_chat(user_input, "en")
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


# --- Tool definitions ---

@tool
def greetings(user_input: str) -> str:
    """Handle greeting intents."""
    pipeline = get_pipeline()
    return pipeline.greeting_generator.generate_greeting(user_input, "casual")


@tool
def out_of_scope(user_input: str) -> str:
    """Handle out-of-scope intents."""
    pipeline = get_pipeline()
    return pipeline.out_of_scope_handler.handle_out_of_scope(user_input)


tools = [greetings, out_of_scope]

# --- Build Graph ---
builder = StateGraph(MessagesState)

# Fixed: node names must match what's used in add_edge/add_conditional_edges
builder.add_node("translate_input", translate_node)
builder.add_node("intent_classifier", classifier_node)
builder.add_node("tools", ToolNode(tools))

# Fixed: added START edge and corrected node name references
builder.add_edge(START, "translate_input")
builder.add_edge("translate_input", "intent_classifier")
builder.add_conditional_edges("intent_classifier", tools_condition)
builder.add_edge("tools", "intent_classifier")  # Loop back after tool execution

# Compile graph
graph = builder.compile()