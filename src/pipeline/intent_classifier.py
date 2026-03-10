"""
Intent Classification Agent
Replaces static pattern matching with LLM-powered intent detection
"""

import logging
from typing import Literal, Optional
from enum import Enum
from langchain_ollama import ChatOllama
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
    secondary_intent: Optional[IntentType] = Field(default=None, description="Secondary intent if message has multiple purposes")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level of classification")
    extracted_query: Optional[str] = Field(default=None, description="Full user query for tool use")
    greeting_type: Optional[str] = Field(default=None, description="Type of greeting: formal, casual, time_based, farewell, gratitude")
    requires_tool_call: bool = Field(default=False, description="Whether this request needs a tool call")  # Fix: was requires_data_access
    tool_to_use: Optional[str] = Field(default=None, description="Tool to use: Policy_RAG_Implementation or zoho_mail")  # Fix: added missing field
    reasoning: str = Field(description="Brief explanation of classification decision")


class IntentClassifier:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.structured_output = llm.with_structured_output(IntentClassification)
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

INTENTS & TOOLS:
- greeting → no tool (ONLY pure greetings with zero question content: hello/bye/thanks/typos like "hiii","gud morning")
- mail_releted → zoho_mail tool
- policy_general_query → Policy_RAG_Implementation tool (policies/rules/person/role/HR/company queries)
- out_of_scope → no tool (unrelated to HR/company)
- unclear → no tool (gibberish/vague)

STRICT CLASSIFICATION RULES:
1. "who is ..." → ALWAYS policy_general_query, NEVER greeting
2. "who are ..." → ALWAYS policy_general_query, NEVER greeting
3. Any job title/role (HR, manager, CEO, head, director, lead) → ALWAYS policy_general_query
4. Any policy/rule/procedure question → ALWAYS policy_general_query
5. Any employee data/leave/attendance question → ALWAYS data_query
6. greeting ONLY when message has NO question and NO request whatsoever
7. greeting + question → primary: policy_general_query or data_query, secondary: greeting
8. requires_tool_call: true for data_query and policy_general_query, ALWAYS
9. extracted_query must NEVER be null when requires_tool_call is true

OUTPUT: Valid YAML only, no markdown:
primary_intent: greeting|data_query|policy_general_query|out_of_scope|unclear
secondary_intent: null|greeting|data_query|policy_general_query
confidence: high|medium|low
extracted_query: <full user query for tool use, NEVER null when requires_tool_call is true>
greeting_type: formal|casual|time_based|farewell|gratitude|null
requires_tool_call: true|false
tool_to_use: Policy_RAG_Implementation|zoho_mail|null
reasoning: <1 sentence>

EXAMPLES:

Input: "who is the HR head"
primary_intent: policy_general_query
secondary_intent: null
confidence: high
extracted_query: who is the HR head
greeting_type: null
requires_tool_call: true
tool_to_use: Policy_RAG_Implementation
reasoning: "who is" with job title always routes to policy_general_query.

Input: "Hi! Who is the HR head?"
primary_intent: policy_general_query
secondary_intent: greeting
confidence: high
extracted_query: Who is the HR head?
greeting_type: null
requires_tool_call: true
tool_to_use: Policy_RAG_Implementation
reasoning: Mixed greeting and person query, primary intent is policy_general_query.

Input: "what is the leave policy?"
primary_intent: policy_general_query
secondary_intent: null
confidence: high
extracted_query: what is the leave policy?
greeting_type: null
requires_tool_call: true
tool_to_use: Policy_RAG_Implementation
reasoning: Company policy question requires Policy_RAG_Implementation tool.

Input: "what's the weather?"
primary_intent: out_of_scope
secondary_intent: null
confidence: high
extracted_query: null
greeting_type: null
requires_tool_call: false
tool_to_use: null
reasoning: Weather is unrelated to HR or company policy.

Input: "asdfgh"
primary_intent: unclear
secondary_intent: null
confidence: high
extracted_query: null
greeting_type: null
requires_tool_call: false
tool_to_use: null
reasoning: Gibberish with no meaningful intent."""

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
                reasoning=f"Classification failed: {str(e)}"
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
    def __init__(self, llm: ChatOllama):
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
            return response.content.strip() if hasattr(response, 'content') else str(response).strip()
        except Exception as e:
            logger.error(f"Greeting generation failed: {e}", exc_info=True)
            return (
                "Hello! I'm MACOM AI Assistant. I can help you with employee records, "
                "leave data, and HR information. What would you like to know?"
            )


class OutOfScopeHandler:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def handle_out_of_scope(self, user_input: str) -> str:
        return (
            f"I appreciate your message, but I'm specifically designed to help with "
            "employee and HR data analysis. I can assist with employee records, leave "
            "balances, department information, and HR-related queries. What would you "
            "like to know about our employee data?"
        )


class UnclearHandler:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def handle_unclear(self, user_input: str) -> str:
        return (
            "I'm not quite sure what you're looking for. I specialize in employee and HR data analysis. "
            "Could you please rephrase your question? For example, you can ask about employee details, "
            "leave records, department information, or any HR-related data."
        )