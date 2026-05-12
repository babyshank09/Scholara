import streamlit as st
import os
from src.backend.config.config import PROJECT_ROOT
from IPython.display import Image
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def format_chat_history(messages):
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
        else:
            formatted.append(msg.content)
    return "\n".join(formatted) 


@st.cache_resource()
def display_langgraph_image(_graph):
    output_file_path = os.path.join(PROJECT_ROOT, "debugging/graph_workflow.png")
    try:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        Image(_graph.get_graph().draw_mermaid_png(output_file_path=output_file_path))
        with st.expander("📊 View LangGraph Workflow"):
            st.image(output_file_path, caption="LangGraph Workflow Visualization")
    except Exception as e:
        st.error(f"Could not generate graph image: {e}")


def display_chat_history(history):
    for msg in history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)