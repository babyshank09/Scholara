import os
import uuid
import asyncio
import json
from dotenv import load_dotenv

from src.backend.graph.graph import build_eval_workflow
from src.backend.llm.llm import get_eval_llm
from src.backend.rag.ingestion_pipeline import IngestionPipeline
from src.backend.config.logging_config import logging, DEBUG

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma




load_dotenv()

CURRENT_FILE = os.path.abspath(__file__)
ABST_TEST_ROOT = os.path.dirname(CURRENT_FILE)
ABST_QUERIES_PATH = os.path.join(ABST_TEST_ROOT, "abstention_queries.json")
SEM = asyncio.Semaphore(10) 




def invoke_graph(app: StateGraph, query: str, llm, vectorstore: Chroma) -> dict:
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
            "vectorstore": vectorstore,
            "llm": llm,
        }
    }

    result = app.invoke(
        input={"messages": [HumanMessage(content=query)]},
        config=config,
    )

    raw_docs = result.get("retrieved_docs", [])
    print(f"raw_docs_len = {len(raw_docs)}")
    retrieved_chunks = [
        {
            "page_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}")),
                "source": doc.metadata.get("source", ""),
            }
        }
        for doc in raw_docs
    ]

    return {
        "query":              query,
        "generated_response": result.get("messages", [])[-1].content if result.get("messages") else "",
        "retrieved_chunks":   retrieved_chunks,
        "thread_id":          thread_id,
    }



async def run_eval_graph_task(app: StateGraph, query: str, llm, vectorstore: Chroma) -> dict:
    async with SEM:
        return await asyncio.to_thread(
            invoke_graph,
            app=app,
            query=query,
            llm=llm,
            vectorstore=vectorstore,
        )



async def generate_abstention_results(queries_path: str, output_path: str, abstention_dataset_path: str, vectorstore: Chroma):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("CO_API_KEY")

    llm = get_eval_llm(openai_key=openai_api_key)
    app = build_eval_workflow(
        openai_api_key=openai_api_key,
        cohere_api_key=cohere_api_key,
    )

    with open(queries_path, "r", encoding="utf-8") as f:
        content = json.load(f)
        queries = content.get("queries", [])

    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(
                run_eval_graph_task(app=app, query=query, llm=llm, vectorstore=vectorstore)
            )
            for query in queries
        ]

    results = []
    for task in tasks:
        task_result = task.result()
        results.append(task_result)

    abstention_eval_result = {
        "queries":            [r["query"] for r in results],
        "generated_responses": [r["generated_response"] for r in results],
        "retrieved_chunks":   [[chunk["metadata"]["original_content"]["raw_text"] for chunk in r["retrieved_chunks"]] for r in results],
        "thread_ids":         [r["thread_id"] for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(abstention_dataset_path, "w", encoding="utf-8") as f:
        json.dump(abstention_eval_result, f, indent=4, ensure_ascii=False) 

    print(f"Saved {len(results)} results → {output_path}")
    print(f"Abstention eval results saved → {abstention_dataset_path}")



def run_abstention_eval_pipeline(output_path: str, abstention_dataset_path: str):
    openai_api_key = os.getenv("OPENAI_API_KEY")

    ingestion_pipeline = IngestionPipeline(log= DEBUG, openai_api_key= openai_api_key, eval_mode= True)
    vectorstore = ingestion_pipeline.run_ingestion_pipeline()

    asyncio.run(generate_abstention_results(
        queries_path= ABST_QUERIES_PATH,
        output_path= OUTPUT_PATH,
        abstention_dataset_path= ABST_DATASET_PATH,
        vectorstore=vectorstore,
    )) 


if __name__ == "__main__":
    OUTPUT_PATH = os.path.join(ABST_TEST_ROOT, "abstention_results.json")
    ABST_DATASET_PATH = os.path.join(ABST_TEST_ROOT, "abstention_dataset.json")

    run_abstention_eval_pipeline(output_path= OUTPUT_PATH, abstention_dataset_path= ABST_DATASET_PATH)