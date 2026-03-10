from langchain_core.tools import tool

@tool
def zoho_mail(user_input: str) -> str:
    """Use ONLY when the user explicitly asks to send an email..."""
    from singleton import get_pipeline
    pipeline = get_pipeline()
    return pipeline.Analyzer.analyze(user_input)
tools = [zoho_mail] 