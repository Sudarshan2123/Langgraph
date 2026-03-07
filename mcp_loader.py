from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
import logging

# Toggle: set to "sse" or "stdio"
TRANSPORT = "stdio"

MCP_SERVER_PATH = "E:\\Agentic_Chatbot\\Mcp"
MCP_SSE_URL = "http://localhost:3001/sse"

_tools = None
_client_ctx = None
_session_ctx = None


async def get_mcp_tools():
    global _tools, _client_ctx, _session_ctx

    if _tools is not None:
        return _tools

    try:
        if TRANSPORT == "stdio":
            server_params = StdioServerParameters(
                command="python",
                args=["mcp_server.py"],
                cwd=MCP_SERVER_PATH
            )
            _client_ctx = stdio_client(server_params)
        else:
            _client_ctx = sse_client(MCP_SSE_URL)

        read, write = await _client_ctx.__aenter__()

        _session_ctx = ClientSession(read, write)
        session = await _session_ctx.__aenter__()

        await session.initialize()
        _tools = await load_mcp_tools(session)
        logging.info(f"[{TRANSPORT}] MCP tools loaded: {[t.name for t in _tools]}")
        return _tools

    except Exception as e:
        logging.warning(f"MCP server unavailable ({TRANSPORT}): {e}")
        _tools = []
        return _tools


async def close_mcp_session():
    global _session_ctx, _client_ctx
    try:
        if _session_ctx:
            await _session_ctx.__aexit__(None, None, None)
        if _client_ctx:
            await _client_ctx.__aexit__(None, None, None)
    except Exception as e:
        logging.warning(f"Error closing MCP session: {e}")