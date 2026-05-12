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
from langchain_openai import OpenAIEmbeddings

from datasets import Dataset
import ragas 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper





load_dotenv()

CURRENT_FILE = os.path.abspath(__file__)
RAGAS_TEST_ROOT = os.path.dirname(CURRENT_FILE)
QA_PAIR_PATH = os.path.join(RAGAS_TEST_ROOT, "question_answer_pairs.json")
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
                "source":           doc.metadata.get("source", ""),
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



async def generate_eval_dataset(qa_pair_path: str, output_path: str, ragas_eval_dataset_path: str, vectorstore: Chroma):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("CO_API_KEY")

    llm = get_eval_llm(openai_key=openai_api_key)
    app = build_eval_workflow(
        openai_api_key=openai_api_key,
        cohere_api_key=cohere_api_key,
    )

    with open(qa_pair_path, "r", encoding="utf-8") as f:
        content = json.load(f)
        queries = content.get("queries", [])
        reference_answers = content.get("reference_answers", [])

    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(
                run_eval_graph_task(app=app, query=query, llm=llm, vectorstore=vectorstore)
            )
            for query in queries
        ]

    results = []
    for task, reference_answer in zip(tasks, reference_answers):
        task_result = task.result()
        task_result["reference_answer"] = reference_answer
        results.append(task_result)

    ragas_eval_dataset = {
        "queries":            [r["query"] for r in results],
        "reference_answers":  [r["reference_answer"] for r in results],
        "generated_responses": [r["generated_response"] for r in results],
        "retrieved_chunks":   [[chunk["metadata"]["original_content"]["raw_text"] for chunk in r["retrieved_chunks"]] for r in results],
        "thread_ids":         [r["thread_id"] for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(ragas_eval_dataset_path, "w", encoding="utf-8") as f:
        json.dump(ragas_eval_dataset, f, indent=4, ensure_ascii=False) 

    print(f"Saved {len(results)} results → {output_path}")
    print(f"RAGAS dataset saved → {ragas_eval_dataset_path}")



def load_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Dataset.from_dict({
        "user_input": data["queries"],
        "response": data["generated_responses"],
        "reference": data["reference_answers"],
        "retrieved_contexts": data["retrieved_chunks"],
    })



def run_ragas_eval_pipeline(output_path: str, ragas_eval_dataset_path: str ):
    openai_api_key = os.getenv("OPENAI_API_KEY")

    ingestion_pipeline = IngestionPipeline(log= DEBUG, openai_api_key= openai_api_key, eval_mode= True)
    vectorstore = ingestion_pipeline.run_ingestion_pipeline()

    asyncio.run(generate_eval_dataset(
        qa_pair_path=QA_PAIR_PATH,
        output_path= output_path,
        ragas_eval_dataset_path= ragas_eval_dataset_path,
        vectorstore=vectorstore,
    )) 

    llm = LangchainLLMWrapper(
        get_eval_llm(openai_key = openai_api_key)
    )

    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model = "text-embedding-3-small", openai_api_key = openai_api_key) 
    )

    dataset = load_dataset(path = os.path.join(RAGAS_TEST_ROOT, "ragas_eval_dataset.json"))
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    

    df = result.to_pandas()

    ragas_results_filepath = os.path.join(RAGAS_TEST_ROOT, "ragas_results.json")
    df.to_json(ragas_results_filepath, orient="records", indent=4)

    aggregate = {
        "faithfulness":       round(df["faithfulness"].mean(), 4),
        "answer_relevancy":   round(df["answer_relevancy"].mean(), 4),
        "context_precision":  round(df["context_precision"].mean(), 4),
        "context_recall":     round(df["context_recall"].mean(), 4),
    }
    aggregate_filepath = os.path.join(RAGAS_TEST_ROOT, "ragas_aggregate_scores.json")
    with open(aggregate_filepath, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=4)

    print(f"✅ Per-row results → {ragas_results_filepath}")
    print(f"✅ Aggregate scores → {aggregate_filepath}")

    
   




if __name__ == "__main__":
    OUTPUT_PATH = os.path.join(RAGAS_TEST_ROOT, "generated_results.json")
    RAGAS_EVAL_DATASET_PATH = os.path.join(RAGAS_TEST_ROOT, "ragas_eval_dataset.json") 

    run_ragas_eval_pipeline(output_path = OUTPUT_PATH, ragas_eval_dataset_path= RAGAS_EVAL_DATASET_PATH)













