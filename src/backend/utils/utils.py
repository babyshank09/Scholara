import os 
import asyncio 
import atexit
import threading 
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from src.backend.config.logging_config import logging, DEBUG
from src.backend.config.config import PROJECT_ROOT
from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage



UPLOAD_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if DEBUG else logging.WARNING)


@st.cache_resource
def get_background_loop():
    """A plain asyncio loop in a daemon thread — survives all Streamlit reruns."""
    loop = asyncio.new_event_loop()

    def run():
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    atexit.register(loop.stop)
    return loop


def run_async(coro):
    """Submit a coroutine to the background loop, carrying the Streamlit session context."""
    loop = get_background_loop()
    ctx = get_script_run_ctx()

    async def _with_ctx():
        add_script_run_ctx(threading.current_thread(), ctx)
        return await coro

    future = asyncio.run_coroutine_threadsafe(_with_ctx(), loop)
    return future.result()


def add_files_to_docs(files):
    for file in files:
        if file.name in get_current_files_in_docs():
            logger.info("File already exists in docs directory, skipping: %s", file.name)
            continue
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    logger.info("Added %d files to docs directory: %s", len(files), [file.name for file in files])


def get_current_files_in_docs():
    filenames = os.listdir(os.path.join(PROJECT_ROOT, "docs"))
    logger.info(f"Check for files in docs directory: {filenames}")
    return filenames


def load_chat_history(workflow: StateGraph):
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    snapshot = run_async(workflow.aget_state(config))
    intro_msg = AIMessage(content="Hey there! I'm Scholara. Upload a research papers and you can start asking me anything about it!")
    return snapshot.values.get("messages", [intro_msg])

