from langchain_core.tools import tool



# --- Tool definitions ---



@tool
def greetings(user_input: str) -> str:
    """Handle greeting intents."""
    from singleton import get_pipeline
    pipeline = get_pipeline()
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
    return pipeline.mixed_handler.handle_mixed(user_input)


tools = [greetings, out_of_scope, unclear]