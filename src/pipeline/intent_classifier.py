"""
Intent Classification Agent
Replaces static pattern matching with LLM-powered intent detection
"""

import logging
from typing import Literal, Optional
from enum import Enum
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    GREETING = "greeting"
    MAIL = "MAIL_QUERY"
    POLICY_RAG_RETRIVAL = "policy_general_query"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"


class IntentClassification(BaseModel):
    primary_intent: IntentType = Field(description="Primary intent of the user message")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level of classification")
    extracted_query: Optional[str] = Field(default=None, description="Full user query for tool use")
    greeting_type: Optional[str] = Field(default=None, description="Type of greeting: formal, casual, time_based, farewell, gratitude")
    requires_tool_call: bool = Field(default=False, description="Whether this request needs a tool call")


class IntentClassifier:
    def __init__(self, llm: ChatNVIDIA):
        self.llm = llm
        self.structured_output = llm.with_structured_output(IntentClassification,method="json_mode")
        self._init_classifier_chain()
        self._classification_cache = {}
        self._cache_max_size = 100
        logger.info("IntentClassifier initialized")

    def _init_classifier_chain(self):
        system_prompt = self._get_classification_prompt()
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_input}")
        ])
        self.classifier_chain = prompt_template | self.structured_output

    def _get_classification_prompt(self) -> str:
        return """You are an intent classifier for MACOM AI Assistant (HR chatbot).

INTENTS:
- greeting: ONLY pure greetings with zero question content (hello/bye/thanks/typos like "hiii", "gud morning")
- MAIL_QUERY: anything related to email or mail
- policy_general_query: policies/rules/person/role/HR/company queries
- out_of_scope: unrelated to HR/company
- unclear: gibberish/vague

STRICT CLASSIFICATION RULES:
1. Any job title/role (HR, manager, CEO, head, director, lead) → ALWAYS policy_general_query
2. Any policy/rule/procedure question → ALWAYS policy_general_query
3. greeting ONLY when message has NO question and NO request whatsoever
4. greeting + question → primary: policy_general_query
5. requires_tool_call: true for MAIL_QUERY and policy_general_query ALWAYS
6. extracted_query must NEVER be null when requires_tool_call is true

You MUST respond with a valid JSON object only. No explanation, no markdown, no extra text.

EXAMPLES:

Input: "who is the HR head"
{{"primary_intent": "policy_general_query", "confidence": "high", "extracted_query": "who is the HR head", "greeting_type": null, "requires_tool_call": true}}

Input: "what is the leave policy?"
{{"primary_intent": "policy_general_query", "confidence": "high", "extracted_query": "what is the leave policy?", "greeting_type": null, "requires_tool_call": true}}

Input: "hello"
{{"primary_intent": "greeting", "confidence": "high", "extracted_query": null, "greeting_type": "casual", "requires_tool_call": false}}

Input: "good morning"
{{"primary_intent": "greeting", "confidence": "high", "extracted_query": null, "greeting_type": "time_based", "requires_tool_call": false}}

Input: "what's the weather?"
{{"primary_intent": "out_of_scope", "confidence": "high", "extracted_query": null, "greeting_type": null, "requires_tool_call": false}}

Input: "asdfgh"
{{"primary_intent": "unclear", "confidence": "high", "extracted_query": null, "greeting_type": null, "requires_tool_call": false}}
"""

    def clear_cache(self):
        self._classification_cache.clear()
        logger.info("Classification cache cleared")

    def classify_intent(self, user_input: str) -> IntentClassification:
        cache_key = user_input.lower().strip()

        if cache_key in self._classification_cache:
            return self._classification_cache[cache_key]
        try:
            classification = self.classifier_chain.invoke({"user_input": user_input})

            # Ensure extracted_query is populated when tool call is required
            if classification.requires_tool_call and not classification.extracted_query:
                classification.extracted_query = user_input

            if len(self._classification_cache) >= self._cache_max_size:
                self._classification_cache.pop(next(iter(self._classification_cache)))

            self._classification_cache[cache_key] = classification
            return classification

        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)
            return IntentClassification(
                primary_intent=IntentType.UNCLEAR,
                confidence="low",
                requires_tool_call=False,
            )

    def should_skip_table_routing(self, classification: IntentClassification) -> tuple[bool, Optional[str]]:
        if classification.primary_intent == IntentType.GREETING:
            return True, "greeting_detected"
        if classification.primary_intent == IntentType.OUT_OF_SCOPE:
            return True, "out_of_scope"
        if classification.primary_intent == IntentType.UNCLEAR and classification.confidence == "low":
            return True, "unclear_intent"
        return False, None


class GreetingGenerator:
    def __init__(self, llm: ChatNVIDIA):
        self.llm = llm
        self._init_generator_chain()
        logger.info("GreetingGenerator initialized")

    def _init_generator_chain(self):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are MACOM AI Assistant, a friendly HR chatbot.
Generate a warm, concise greeting response appropriate for the greeting type.
Keep it short (1-2 sentences) and professional.
Do not mention data or queries — just respond to the greeting naturally."""),
            ("human", "User said: {user_input}\nGreeting type: {greeting_type}")
        ])
        self.generator_chain = prompt_template | self.llm

    def generate_greeting(self, user_input: str, greeting_type: str) -> str:
        try:
            response = self.generator_chain.invoke({
                "user_input": user_input,
                "greeting_type": greeting_type
            })
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, dict):
            # ChatNVIDIA can return a dict with a 'content' or 'text' key
                return (response.get('content') or response.get('text') or str(response)).strip()
            return str(response).strip()
        except Exception as e:
            logger.error(f"Greeting generation failed: {e}", exc_info=True)
            return (
                "Hello! I'm MACOM AI Assistant. I can help you with employee records, "
                "leave data, and HR information. What would you like to know?"
            )


class OutOfScopeHandler:
    def __init__(self, llm: ChatNVIDIA):
        self.llm = llm

    def handle_out_of_scope(self, user_input: str) -> str:
        return (
            f"I appreciate your message, but I'm specifically designed to help with "
            "employee and HR data analysis. I can assist with employee records, leave "
            "balances, department information, and HR-related queries. What would you "
            "like to know about our employee data?"
        )


class UnclearHandler:
    def __init__(self, llm: ChatNVIDIA):
        self.llm = llm

    def handle_unclear(self, user_input: str) -> str:
        return (
            "I'm not quite sure what you're looking for. I specialize in employee and HR data analysis. "
            "Could you please rephrase your question? For example, you can ask about employee details, "
            "leave records, department information, or any HR-related data."
        )