from typing import Annotated
from langchain_core.messages import BaseMessage
import operator
from src.pipeline.intent_classifier import IntentClassification
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State management for the AI agent."""
    messages : Annotated[list[BaseMessage], operator.add]
    classification : IntentClassification