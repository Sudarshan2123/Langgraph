"""
Intent Classification Agent
Replaces static pattern matching with LLM-powered intent detection
"""

import logging
from typing import  Literal, Optional
from enum import Enum
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
# from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Types of user intents"""
    GREETING = "greeting"
    DATA_QUERY = "data_query"
    MIXED = "mixed"  # Greeting + Query
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"


class IntentClassification(BaseModel):
    """Structured intent classification result"""
    primary_intent: IntentType = Field(
        description="Primary intent of the user message"
    )
    secondary_intent: Optional[IntentType] = Field(
        default=None,
        description="Secondary intent if message has multiple purposes"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of classification"
    )
    extracted_query: Optional[str] = Field(
        default=None,
        description="For MIXED intent, the extracted data query portion"
    )
    greeting_type: Optional[str] = Field(
        default=None,
        description="Type of greeting: formal, casual, time_based, farewell, gratitude"
    )
    requires_data_access: bool = Field(
        description="Whether this request needs database access"
    )
    reasoning: str = Field(
        description="Brief explanation of classification decision"
    )


class IntentClassifier:
    """
    LLM-powered intent classifier that handles:
    - Greetings (with typos)
    - Data queries
    - Mixed intents
    - Out-of-scope requests
    """

    def __init__(self, llm: ChatOllama):
        """
        Initialize intent classifier.

        Args:
            llm: VertexAI LLM instance
        """
        self.llm = llm
        self.structured_output = llm.with_structured_output(IntentClassification)
        self._init_classifier_chain()

        # 🔧 Add cache for common inputs
        self._classification_cache = {}
        self._cache_max_size = 100

        # Clear cache on initialization to ensure fresh classifications
        self._classification_cache.clear()

        logger.info("IntentClassifier initialized")

    def _init_classifier_chain(self):
        """Initialize the classification chain"""
        system_prompt = self._get_classification_prompt()

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_input}")
        ])

        self.classifier_chain = prompt_template | self.llm 

    def _get_classification_prompt(self) -> str:
        """Get the system prompt for intent classification"""
        return """You are an intent classification expert for an employee data analysis chatbot called MACOM AI Assistant.

**YOUR TASK**: Analyze the user's message and classify their intent.

**AVAILABLE INTENTS**:

1. **GREETING**:
   - User is saying hello, goodbye, or expressing gratitude
   - Includes typos and variations: "helo", "hiii", "hiiii", "gud morning", "thanx", "bye bye"
   - Examples: "hi", "hiii", "hello there", "good morning!", "thanks", "goodbye"
   - Types: formal, casual, time_based, farewell, gratitude

2. **DATA_QUERY**:
   - User wants to retrieve, analyze, or explore employee/HR data
   - Examples: "show me employee details", "who works in engineering", "leave balance"
   - Requires database access

3. **MIXED**:
   - Greeting + Data query in same message
   - Examples: "Hi! Can you show me employee list?", "Good morning, what's my leave balance?"
   - Extract the query portion for processing

4. **OUT_OF_SCOPE**:
   - Requests unrelated to HR/employee data
   - Examples: "what's the weather", "tell me a joke", "write me a poem"
   - Politely decline these

5. **UNCLEAR**:
   - Ambiguous, too vague, or nonsensical input
   - Examples: "asdfgh", "???", "help"

**CRITICAL RULES**:
- Be VERY tolerant of typos and variations (up to 3 character errors per word)
- "helo" → GREETING
- "hiii" → GREETING 
- "hiiii" → GREETING
- "gud mornign" → GREETING
- "show employe detals" → DATA_QUERY (tolerate typos)
- Mixed intents should extract ONLY the query portion
- If unsure between GREETING and DATA_QUERY, choose DATA_QUERY (better to try than reject)

**OUTPUT FORMAT**:
Respond with ONLY valid YAML (no markdown code blocks, no preamble, no explanations):

primary_intent: "greeting" | "data_query" | "mixed" | "out_of_scope" | "unclear"
secondary_intent: null | "greeting" | "data_query"
confidence: "high" | "medium" | "low"
extracted_query: "only for mixed intent - the data query part"
greeting_type: "formal" | "casual" | "time_based" | "farewell" | "gratitude" | null
requires_data_access: true | false
reasoning: "1 sentence explanation"

**EXAMPLES**:

Input: "hiii"
Output:
primary_intent: greeting
secondary_intent: null
confidence: high
extracted_query: null
greeting_type: casual
requires_data_access: false
reasoning: Casual greeting with repeated letters common in informal communication.

Input: "Good morning, show me the leave balance for John Doe"
Output:
primary_intent: mixed
secondary_intent: data_query
confidence: high
extracted_query: show me the leave balance for John Doe
greeting_type: time_based
requires_data_access: true
reasoning: Message contains both a time-based greeting and a specific HR data request."""

    def clear_cache(self):
        """Clear the classification cache"""
        self._classification_cache.clear()
        logger.info("Classification cache cleared")

    def classify_intent(self, user_input: str) -> IntentClassification:
        """
        Classify user intent with retry logic.

        Args:
            user_input: User's message

        Returns:
            IntentClassification object
        """
        # Check cache first
        cache_key = user_input.lower().strip()

        if cache_key in self._classification_cache:
            return self._classification_cache[cache_key]
        try:
            classification = self.classifier_chain.invoke({"user_input": user_input})
            # classification = IntentClassification(**result)

             # Cache the result
            if len(self._classification_cache) >= self._cache_max_size:
                # Remove oldest entry
                self._classification_cache.pop(next(iter(self._classification_cache)))

            self._classification_cache[cache_key] = classification

            return classification

        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)

            # Fallback classification
            return IntentClassification(
                primary_intent=IntentType.UNCLEAR,
                confidence="low",
                requires_data_access=False,
                reasoning=f"Classification failed: {str(e)}"
            )

    def should_skip_table_routing(
        self,
        classification: IntentClassification
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if table routing should be skipped.

        Args:
            classification: Intent classification result

        Returns:
            Tuple of (should_skip, reason)
        """
        if classification.primary_intent == IntentType.GREETING:
            return True, "greeting_detected"

        if classification.primary_intent == IntentType.OUT_OF_SCOPE:
            return True, "out_of_scope"

        if classification.primary_intent == IntentType.UNCLEAR:
            if classification.confidence == "low":
                return True, "unclear_intent"

        # For MIXED and DATA_QUERY, proceed to table routing
        return False, None


class GreetingGenerator:
    """
    Generates dynamic, contextual greeting responses.
    """

    def __init__(self, llm: ChatOllama):
        """
        Initialize greeting generator.

        Args:
            llm: VertexAI LLM instance
        """
        self.llm = llm
        self._init_generator_chain()

        logger.info("GreetingGenerator initialized")

    def _init_generator_chain(self):
        """Initialize the greeting generation chain"""
        system_prompt = self._get_generation_prompt()

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User said: {user_input}\nGreeting type: {greeting_type}")
        ])

        self.generator_chain = prompt_template | self.llm

    def _get_generation_prompt(self) -> str:
        """Get the system prompt for greeting generation"""
        return """You are MACOM AI Assistant, a friendly and professional employee data analysis chatbot.

**YOUR TASK**: Generate a warm, natural greeting response.

**PERSONALITY TRAITS**:
- Professional but approachable
- Enthusiastic about helping with employee data
- Concise (2-3 sentences max)
- Adaptive to user's tone

**RESPONSE GUIDELINES**:

1. **Casual Greetings** (hi, hey, hello):
   - "Hello! I'm MACOM AI Assistant. I can help you analyze employee records, leave data, and HR information. What would you like to know?"

2. **Time-based Greetings** (good morning, good evening):
   - Mirror the time reference: "Good morning! Ready to help you..."
   - Add slight variation: "Good afternoon! I'm here to assist with..."

3. **Formal Greetings** (greetings, good day):
   - Match formality: "Greetings! I am MACOM AI Assistant, your..."

4. **Farewells** (bye, goodbye, see you):
   - Warm closure: "Goodbye! Feel free to return anytime you need help with employee data."
   - Add helpfulness: "Take care! I'm here whenever you need assistance."

5. **Gratitude** (thanks, thank you):
   - Acknowledge: "You're welcome! Happy to help."
   - Encourage return: "My pleasure! Let me know if you need anything else."

**CRITICAL RULES**:
- NEVER mention that you detected a typo
- Keep responses fresh (slight variations each time)
- Always mention you can help with employee/HR/leave data
- Stay within 2-3 sentences
- NO markdown formatting, NO bullet points
- Sound natural, like a helpful colleague"""

    def generate_greeting(
        self,
        user_input: str,
        greeting_type: str
    ) -> str:
        """
        Generate contextual greeting response.

        Args:
            user_input: User's original message
            greeting_type: Type of greeting (casual, formal, time_based, farewell, gratitude)

        Returns:
            Dynamic greeting response
        """
        try:
            response = self.generator_chain.invoke({
                "user_input": user_input,
                "greeting_type": greeting_type
            })

            # Extract text from response
            if hasattr(response, 'content'):
                greeting_text = response.content.strip()
            else:
                greeting_text = str(response).strip()

            logger.info(f"Generated {greeting_type} greeting")

            return greeting_text

        except Exception as e:
            logger.error(f"Greeting generation failed: {e}", exc_info=True)

            # Fallback greeting
            return (
                "Hello! I'm MACOM AI Assistant. I can help you analyze employee records, "
                "leave data, and HR information. What would you like to know?"
            )


class OutOfScopeHandler:
    """
    Handles out-of-scope requests with helpful redirects.
    """

    def __init__(self, llm: ChatOllama):
        """
        Initialize out-of-scope handler.

        Args:
            llm: VertexAI LLM instance
        """
        self.llm = llm

    def handle_out_of_scope(self, user_input: str) -> str:
        """
        Generate helpful response for out-of-scope requests.

        Args:
            user_input: User's message

        Returns:
            Polite redirection message
        """
        return (
            f"I appreciate your message: '{user_input}'. However, I'm specifically designed to help with "
            "employee and HR data analysis. I can assist you with questions about employee records, leave "
            "balances, department information, and other HR-related queries. What would you like to know "
            "about our employee data?"
        )



class UnclaerHandler:
    """Handler Unclear Scope requests with redirects"""

    def __init__(self,llm:ChatOllama):
        """
        initialized unclear scope Handle
        
        Args: 
           llm:Uses Chatollama
        """
        self.llm = llm 

    def handle_unclear(self, user_input: str) -> str:
        """
        Handle unclear or ambiguous input.

        Args:
            user_input: User's message

        Returns:
            Clarification request
        """
        return (
            "I'm not quite sure what you're looking for. I specialize in employee and HR data analysis. "
            "Could you please rephrase your question? For example, you can ask about employee details, "
            "leave records, department information, or any HR-related data."
        )

