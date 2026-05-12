from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import trim_messages
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever

import os
import uuid
import asyncio
import threading
import atexit
import streamlit as st
from IPython.display import Image
from dotenv import load_dotenv
from urllib.parse import quote_plus

from pydantic import BaseModel, Field
from typing import TypedDict, Optional, List, Literal
from typing_extensions import Annotated
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import AnyMessage, add_messages
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pathlib import Path
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from src.backend.config.logging_config import logging, DEBUG
from src.backend.config.config import PROJECT_ROOT
from src.backend.rag.ingestion_pipeline import IngestionPipeline
from src.backend.rag.retrieval_pipeline import RetrievalPipeline
from src.backend.rag.response_synthesis_pipeline import ResponseSynthesisPipeline


# ── Environment & Config ──────────────────────────────────────────────────────

load_dotenv()

DB_URI = (
    f"postgresql://{os.getenv('DB_USER')}:{quote_plus(os.getenv('DB_PASSWORD', ''))}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger("main")
logger.setLevel(logging.INFO if DEBUG else logging.WARNING)


# ── Background Event Loop ─────────────────────────────────────────────────────

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


# ── Streamlit Page Config ─────────────────────────────────────────────────────

st.set_page_config(page_title="Retrievon", page_icon="🌟")
st.title("🌟 Retrievon")
st.markdown("##### Turn your knowledge base into insights with AI-powered RAG")


# ── Helper Functions ──────────────────────────────────────────────────────────

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





# ── Session State Init ────────────────────────────────────────────────────────

if "current_embedded_files" not in st.session_state:
    st.session_state["current_embedded_files"] = []

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

if "thread_id" not in st.session_state:
    # st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["thread_id"] = "static-thread-id-for-dev"  


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.header("🔑 API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))
    cohere_key = st.text_input("Cohere API Key", type="password", value=os.getenv("CO_API_KEY")) 

    if not openai_key or not cohere_key:
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


# ── LLM ───────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_llm(openai_key):
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_key,
        temperature=0,
        stream_usage=True
    )

llm = get_llm(openai_key)


# ── State & Structured Output Models ─────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Optional[str]
    rewritten_query: Optional[str]
    retrieved_docs: Optional[List[Document]]


class RouteDecision(BaseModel):
    route: Literal["rag", "chat"] = Field(
        description="Route the query to either rag or chat"
    )


class QueryRewrite(BaseModel):
    standalone_query: str = Field(
        description="Standalone version of the user's latest question for document retrieval"
    )


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def orchestrator(state: State, config: RunnableConfig):
    user_query = state["messages"][-1].content
    llm = config["configurable"]["llm"]
    router_llm = llm.with_structured_output(RouteDecision)

    system_prompt = """
    You are the routing brain of a Retrieval-Augmented Generation (RAG) application.

    The user can upload private documents such as:
    - resumes
    - reports
    - policies
    - research papers
    - internal documents
    - personal notes
    - project files

    Your job is to decide whether the question should use:
    RAG → search the uploaded documents
    CHAT → answer using general knowledge

    IMPORTANT PRINCIPLE:
    When unsure, ALWAYS choose RAG.

    Route to RAG if the question:
    - Mentions a person that is not a famous public figure
    - Asks about skills, experience, projects, resumes, profiles
    - Mentions "the document", "the file", "the pdf", "uploaded files"
    - Asks about company, student, employee, or internal info
    - Sounds like it could come from private documents
    - Requires specific or personalized information
    - Could possibly be answered from uploaded files

    Route to CHAT only if the question is CLEARLY:
    - General world knowledge (history, science, math, geography)
    - Casual conversation or opinions
    - Something impossible to exist in private documents

    Examples:
    "What are Akshat's skills?" → RAG
    "Summarize the uploaded pdf" → RAG
    "What is machine learning?" → CHAT
    "Who is the CEO of Google?" → CHAT

    Return ONLY one word:
    rag
    chat
    """

    decision = router_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    node_logger = logging.getLogger("orchestrator")
    node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)
    node_logger.info("User Query: %s | Route Decision: %s", user_query, decision.route)

    return {"route": decision.route}


def route_from_orchestrator(state: State) -> str:
    return "query_rewriting_agent" if state["route"] == "rag" else "response_generation_agent"


def query_rewriting_agent(state: State, config: RunnableConfig):
    llm = config["configurable"]["llm"]
    trimmed_history = trim_messages(
        state["messages"], 
        max_tokens = 4000, 
        token_counter = llm, 
        strategy = "last", 
        include_system = True
    )
    formatted_history = format_chat_history(trimmed_history)
    latest_question = state["messages"][-1].content

    structured_llm = llm.with_structured_output(QueryRewrite)

    system_prompt = """
    You are a Query Rewriting Agent for a Retrieval-Augmented Generation (RAG) system.

    Your job is to rewrite the user's latest question into a clear, standalone
    search query that can be understood WITHOUT the conversation history.

    IMPORTANT RULES:
    - Use the conversation history only to add missing context.
    - Resolve pronouns like "it", "they", "this", "that", "the design", etc.
    - Preserve technical terms, product names, and metrics.
    - Do NOT answer the question.
    - Do NOT add new information.
    - Output ONLY the rewritten query as a single sentence.
    """

    user_prompt = f"""
    Conversation History:
    {formatted_history}

    Latest User Question:
    {latest_question}

    Rewritten Standalone Query:
    """

    result = structured_llm.invoke([SystemMessage(content = system_prompt), HumanMessage(content = user_prompt)])

    node_logger = logging.getLogger("query_rewriting_agent")
    node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)
    node_logger.info("Latest Question: %s | Rewritten Query: %s", latest_question, result.standalone_query)

    return {"rewritten_query": result.standalone_query}


def retriever_agent(state: State, config: RunnableConfig):
    vectorstore = config["configurable"]["vectorstore"]
    retrieval_pipeline = RetrievalPipeline(openai_api_key=openai_key, cohere_api_key = cohere_key, vectorstore=vectorstore, log=DEBUG)
    retrieved_documents = retrieval_pipeline.run_retrieval_pipeline(
        query=state["rewritten_query"],
        retriever_type="hybrid",
        vector_retriever_type="similarity_score_threshold"
    )
    return {"retrieved_docs": retrieved_documents}


def response_generation_agent(state: State, config: RunnableConfig):
    llm = config["configurable"]["llm"]
    if state["route"] == "rag":
        response_synthesis_pipeline = ResponseSynthesisPipeline(openai_api_key=openai_key, log=DEBUG)
        final_answer = response_synthesis_pipeline.generate_final_answer(
            chunks=state["retrieved_docs"],
            query=state["rewritten_query"]
        )
        return {"messages": [AIMessage(content=final_answer)]}
    
    system_prompt = """
    You are a professional AI assistant for a Retrieval-Augmented Generation (RAG) application.

    Your capabilities:
    1) You can answer questions using retrieved documents when they are provided.
    2) You can answer general knowledge questions when no documents are involved.
    3) You must clearly separate document-based answers from general knowledge.

    Document usage rules:
    - When document context is provided, treat it as the PRIMARY source of truth.
    - Base your answer strictly on the retrieved context.
    - If the answer is not present in the documents, say:
    "I could not find this in the uploaded documents."
    - Do NOT hallucinate or invent document content.

    General knowledge rules:
    - If no documents are used, answer normally using your general knowledge.
    - Never pretend information came from documents if it did not.

    Safety & honesty:
    - If you are unsure, say you are unsure.
    - Do not fabricate facts, citations, or data.
    - Do not make up file names, numbers, or policies.

    Response style:
    - Be clear, structured, and concise.
    - Prefer bullet points or short paragraphs.
    - Avoid unnecessary fluff.
    - Be helpful and professional.

    Goal:
    Provide the most accurate and trustworthy answer possible while minimizing hallucinations.
    """
    
    trimmed_history = trim_messages(
        state["messages"], 
        max_tokens = 4000, 
        token_counter = llm, 
        strategy = "last", 
        include_system = True
    ) 

    response = llm.invoke([SystemMessage(content=system_prompt)] + trimmed_history)
    return {"messages": [AIMessage(content=response.content)]}




# ── Build Workflow ────────────────────────────────────────────────────────────

@st.cache_resource
def build_workflow():
    async def _init():
        pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        return checkpointer

    checkpointer = run_async(_init())

    graph = StateGraph(State)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("query_rewriting_agent", query_rewriting_agent)
    graph.add_node("retriever_agent", retriever_agent)
    graph.add_node("response_generation_agent", response_generation_agent)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator", route_from_orchestrator,
        {
            "query_rewriting_agent": "query_rewriting_agent",
            "response_generation_agent": "response_generation_agent"
        }
    )
    graph.add_edge("query_rewriting_agent", "retriever_agent")
    graph.add_edge("retriever_agent", "response_generation_agent")
    graph.add_edge("response_generation_agent", END)

    return graph.compile(
        checkpointer=checkpointer, 
    )


# ── Graph Visualization ───────────────────────────────────────────────────────

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


workflow = build_workflow()
display_langgraph_image(workflow)

    

# ── Chat History Display ──────────────────────────────────────────────────────

def load_chat_history():
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    snapshot = run_async(workflow.aget_state(config))
    return snapshot.values.get("messages", [])

history = load_chat_history()  

def display_chat_history(history):
    for msg in history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

display_chat_history(history)



# ── Streaming ─────────────────────────────────────────────────────────────────

async def stream_graph_events(new_message: str, response_placeholder):
    full_response = ""

    config = {
        "run_name": f"Retrievon_{st.session_state['thread_id']}",
        "configurable": { 
            "thread_id": st.session_state["thread_id"],
            "llm": llm,
            "vectorstore": st.session_state.get("vectorstore"),
        }
    }

    async for event in workflow.astream_events(
        input={"messages": [HumanMessage(content=new_message)]},
        version="v2",
        config=config
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == "response_generation_agent"
        ):
            chunk = event["data"]["chunk"].content
            if chunk:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

    response_placeholder.markdown(full_response)
    return full_response


# ── Chat Input ──────────────────────────────────────────────────────────────── 

user_query = st.chat_input("Ask me anything from your document...")

if user_query and user_query.strip():
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()                                      
        full_response = run_async(stream_graph_events(user_query, response_placeholder))  
