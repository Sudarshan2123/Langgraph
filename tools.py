from langchain_core.tools import tool
from requests import Response



# --- Tool definitions ---
@tool
def structure_data(user_input: str) -> str:
    """Handle the user input to query the database for the data and provide the output for future process"""
    from singleton import get_pipeline
    pipeline = get_pipeline()
    conndata = pipeline.Database.get_table_names()
    agent_state = pipeline.config_obj.AgentState(conndata, user_input)
    state_with_intent = analyzer.detect_table_intent(agent_state)
    return pipeline.greeting_generator.generate_greeting(user_input, "casual")


@tool
def out_of_scope(user_input: str) -> str:
    """Handle out-of-scope intents."""
    from singleton import get_pipeline
    pipeline = get_pipeline()
    return pipeline.out_of_scope_handler.handle_out_of_scope(user_input)

@tool
def unclear(user_input:str) -> str:
    """Handle mixed intents."""
    from singleton import get_pipeline
    pipeline = get_pipeline()
    return pipeline.mixed_handler.handle_unclear(user_input)


tools = [greetings, out_of_scope, unclear]