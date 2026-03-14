from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import logging

TRANSPORT = "sse"
MCP_SERVER_PATH = "E:\\Agentic_Chatbot\\Mcp"
MCP_SSE_URL = "http://localhost:8080/sse"

_tools = None
_session = None
_cm_outer = None  # holds the stdio/sse_client context
_cm_inner = None  # holds the ClientSession context
_init_lock = None


def _get_lock():
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


async def _is_session_alive() -> bool:
    """Check if the current session is still usable."""
    global _session
    if _session is None:
        return False
    try:
        await _session.list_tools()
        return True
    except Exception:
        return False


async def init_mcp_session():
    global _tools, _session, _cm_outer, _cm_inner

    # Fast path: session already alive
    if _session is not None and await _is_session_alive():
        return _tools

    async with _get_lock():
        # Re-check inside lock
        if _session is not None and await _is_session_alive():
            return _tools

        # Clean up dead session if any
        await close_mcp_session()

        try:
            if TRANSPORT == "stdio":
                server_params = StdioServerParameters(
                    command="python",
                    args=["mcp_server.py"],
                    cwd=MCP_SERVER_PATH
                )
                # Enter context managers manually so session stays alive
                _cm_outer = stdio_client(server_params)
                read, write = await _cm_outer.__aenter__()
            else:
                _cm_outer = sse_client(MCP_SSE_URL)
                read, write = await _cm_outer.__aenter__()

            _cm_inner = ClientSession(read, write)
            _session = await _cm_inner.__aenter__()
            await _session.initialize()

            _tools = await load_mcp_tools(_session)
            logging.info(f"[{TRANSPORT}] MCP tools loaded: {[t.name for t in _tools]}")

        except Exception as e:
            logging.warning(f"MCP server unavailable ({TRANSPORT}): {e}")
            _tools = []
            _session = None

    return _tools


def get_mcp_tools() -> list:
    return _tools or []


async def close_mcp_session():
    """Cleanly close the persistent session and transport."""
    global _tools, _session, _cm_outer, _cm_inner

    if _cm_inner is not None:
        try:
            await _cm_inner.__aexit__(None, None, None)
        except Exception:
            pass
        _cm_inner = None

    if _cm_outer is not None:
        try:
            await _cm_outer.__aexit__(None, None, None)
        except Exception:
            pass
        _cm_outer = None

    _session = None
    _tools = None
    logging.info("MCP session closed.")