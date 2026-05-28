import streamlit as st
from langgraph.graph import StateGraph, START, END
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
import functools

from src.backend.state.state import State
from src.backend.nodes.nodes import Nodes
from src.backend.config.config import DB_URI
from src.backend.utils.utils import run_async


@st.cache_resource
def build_workflow(openai_api_key: str, cohere_api_key: str):
    nodes = Nodes(openai_api_key = openai_api_key, cohere_api_key = cohere_api_key)
    
    async def _init():
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as saver:
            await saver.setup()

        pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open()
        return AsyncPostgresSaver(pool)

    checkpointer = run_async(_init())

    graph = StateGraph(State)

    graph.add_node("input_guardrails", nodes.input_guardrails)
    graph.add_node("blocking_agent", nodes.blocking_agent)
    graph.add_node("orchestrator", nodes.orchestrator)
    graph.add_node("clarify_agent", nodes.clarify_agent)
    graph.add_node("query_rewriting_agent", nodes.query_rewriting_agent)
    graph.add_node("retriever_agent", nodes.retriever_agent)
    graph.add_node("response_generation_agent", nodes.response_generation_agent)
    graph.add_node("output_guardrails", nodes.output_guardrails)

    graph.add_edge(START, "input_guardrails")
    graph.add_conditional_edges(
        "input_guardrails",
        nodes.route_from_input_guardrails,
        {
            "orchestrator": "orchestrator",
            "blocking_agent": "blocking_agent"
        }
    )
    graph.add_conditional_edges(
        "orchestrator",
        nodes.route_from_orchestrator,
        {
            "query_rewriting_agent": "query_rewriting_agent",
            "response_generation_agent": "response_generation_agent",
            "clarify_agent": "clarify_agent"         
        },
    )
    graph.add_edge("query_rewriting_agent", "retriever_agent")
    graph.add_edge("retriever_agent", "response_generation_agent")
    graph.add_edge("response_generation_agent", "output_guardrails")
    graph.add_conditional_edges(
        "output_guardrails",
        nodes.route_from_output_guardrails,
        {
            "blocking_agent": "blocking_agent",
            "end": END
        }
    )
    graph.add_edge("clarify_agent", END)  # ← ends turn, user replies next
    graph.add_edge("blocking_agent", END)

    return graph.compile(checkpointer=checkpointer)



@functools.lru_cache(maxsize=1)
def build_eval_workflow(openai_api_key: str, cohere_api_key: str):
    nodes = Nodes(openai_api_key = openai_api_key, cohere_api_key = cohere_api_key, eval_mode = True)

    checkpointer = MemorySaver()

    graph = StateGraph(State)

    graph.add_node("orchestrator", nodes.orchestrator)
    graph.add_node("query_rewriting_agent", nodes.query_rewriting_agent)
    graph.add_node("retriever_agent", nodes.retriever_agent)
    graph.add_node("response_generation_agent", nodes.response_generation_agent)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        nodes.route_from_orchestrator,
        {
            "query_rewriting_agent": "query_rewriting_agent",
            "response_generation_agent": "response_generation_agent",
        },
    )
    graph.add_edge("query_rewriting_agent", "retriever_agent")
    graph.add_edge("retriever_agent", "response_generation_agent")
    graph.add_edge("response_generation_agent", END)

    return graph.compile(checkpointer=checkpointer) 



    
