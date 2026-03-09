from langchain_core.tools import tool

@tool
def structure_data(user_input: str) -> str:
    """Query the database for employee data, leave balances, and department information."""
    from singleton import get_pipeline
    pipeline = get_pipeline()
    return pipeline.Analyzer.analyze(user_input)

tools = [structure_data]  # Only real data tools here, not out_of_scope/unclearS