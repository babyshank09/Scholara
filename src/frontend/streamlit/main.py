import os
import uuid
import streamlit as st

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.backend.config.logging_config import DEBUG
from src.backend.rag.ingestion_pipeline import IngestionPipeline
from src.backend.utils.utils import (
    run_async, 
    add_files_to_docs, 
    get_current_files_in_docs, 
    load_chat_history
)
from src.backend.llm.llm import get_llm 
from src.backend.graph.graph import build_workflow
from src.frontend.utils.utils import display_langgraph_image, display_chat_history
 


def load_rag_application():

    st.set_page_config(page_title="Scholara", page_icon="🔍")
    st.title("🔍 Scholara")
    st.markdown("##### The smartest way to read research")


    if "thread_id" not in st.session_state:
        # st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["thread_id"] = "static-thread-id-for-dev"  



    with st.sidebar:

        st.header("🔑 API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))
        co_key = st.text_input("Cohere API Key", type="password", value=os.getenv("CO_API_KEY"))

        if not openai_key or not co_key:
            st.warning("Please enter your API keys to proceed.")
            st.stop()

        ingestion_pipeline = IngestionPipeline(openai_api_key=openai_key, log=DEBUG)

        st.write(" ")
        st.header("📁 File Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF or Word files",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Embed Documents", type="primary", use_container_width=True):
            add_files_to_docs(uploaded_files)
            with st.spinner("Embedding documents... This may take a moment"):
                vectorstore = ingestion_pipeline.run_ingestion_pipeline()
            st.session_state["vectorstore"] = vectorstore
            st.session_state["current_embedded_files"] = ingestion_pipeline.get_embedded_filenames(
                vectorstore=st.session_state.get("vectorstore")
            )
            st.success("Documents embedded successfully!")


        if len(get_current_files_in_docs()) == 0:
            st.warning("No files embedded. Please upload at least one document to proceed.")
            st.stop()

        if "vectorstore" not in st.session_state:
            with st.spinner("Loading vectorstore... Please wait while I load your embedded documents."):
                vectorstore = ingestion_pipeline.run_ingestion_pipeline()
            st.session_state["vectorstore"] = vectorstore

        if "current_embedded_files" not in st.session_state:
            st.session_state["current_embedded_files"] = ingestion_pipeline.get_embedded_filenames(
                vectorstore=st.session_state.get("vectorstore")
            )

        st.write(" ")
        st.header("🗑️ Delete File")
        st.session_state["current_embedded_files"] = ingestion_pipeline.get_embedded_filenames(
            vectorstore=st.session_state.get("vectorstore")
        )
        filename = st.selectbox("Select a file to delete", st.session_state.get("current_embedded_files"))

        if st.button("Delete", type="primary", use_container_width=True):
            ingestion_pipeline.delete_chunks_by_source(
                vectorstore=st.session_state.get("vectorstore"),
                source_filename=filename
            )
            st.session_state["current_embedded_files"] = ingestion_pipeline.get_embedded_filenames(
                vectorstore=st.session_state.get("vectorstore")
            )
            st.error("Deleted file and associated chunks: %s" % filename)

        st.write(" ")
        st.write(" ")
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()


    llm = get_llm(openai_key=openai_key)

    workflow = build_workflow(openai_api_key = openai_key, cohere_api_key = co_key) 
    display_langgraph_image(workflow)

    history = load_chat_history(workflow)  
    display_chat_history(history)



    async def stream_graph_events(new_message: str, response_placeholder):
        full_response = ""

        config = {
            "run_name": f"Scholara_{st.session_state['thread_id']}",
            "configurable": { 
                "thread_id": st.session_state["thread_id"],
                "llm": llm,
                "vectorstore": st.session_state.get("vectorstore"), 
                "current_embedded_files": st.session_state["current_embedded_files"]
            }
        }

        async for event in workflow.astream_events(
            input={"messages": [HumanMessage(content=new_message)]},
            version="v2",
            config=config
        ):
            if (
                event["event"] == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") in ("response_generation_agent", "blocking_agent", "clarify_agent")
            ):
                chunk = event["data"]["chunk"].content
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        return full_response




    user_query = st.chat_input("Ask me anything from your document...")

    if user_query and user_query.strip():
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()                                      
            full_response = run_async(stream_graph_events(user_query, response_placeholder))  

























