import json
import os 
import logging
from typing import List 
from collections import defaultdict

from langchain_core.documents import Document
from langchain_chroma import Chroma
# from langchain_classic.retrievers import EnsembleRetriever
from langchain.retrievers import EnsembleRetriever  
from langchain_community.retrievers import BM25Retriever 
from langchain_cohere import CohereRerank
from langsmith import traceable

from pydantic import BaseModel, Field 
from dotenv import load_dotenv  

from src.backend.rag.ingestion_pipeline import IngestionPipeline
from src.backend.config.logging_config import logging, DEBUG
from src.backend.llm.llm import get_llm, get_eval_llm



load_dotenv() 

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))


class QueryVariations(BaseModel):
    queries: List[str] = Field(
        min_items=5, 
        max_items=5,
        description="""
            Exactly five standalone search queries that preserve the same meaning as the user's question.
            The first query must be the original user query unchanged, and the remaining four must be
            semantically equivalent variations optimized for search retrieval.
        """
    )

 
class RetrievalPipeline:  

    def __init__(self, openai_api_key: str, cohere_api_key: str, vectorstore: Chroma, log: bool = False, eval_mode: bool = False): 
        self.log = log
        self.logger = logging.getLogger(self.__class__.__name__) 
        self.logger.setLevel(logging.INFO if log else logging.WARNING)  
        self.vectorstore = vectorstore
        self.reranking_query = None 

        if eval_mode:
            self.logger.info("Initializing RetrievalPipeline in EVAL MODE with detailed logging enabled.")
            self.llm = get_eval_llm(openai_key = openai_api_key)

        else:
            self.llm = get_llm(openai_key = openai_api_key)

        self.reranker = CohereRerank(model = "rerank-english-v3.0", top_n= 3, cohere_api_key=cohere_api_key) 



    def generate_query_variations(self, original_query):
        import time
        from openai import RateLimitError

        self.logger.info("Generating query variations using LLM...")
        self.logger.info(f"Original Query: {original_query}\n")
        self.reranking_query = original_query  

        query_llm = self.llm.with_structured_output(QueryVariations)

        prompt = f"""
        Generate 4 different variations of this query that would help retrieve relevant documents:
        Original query: {original_query}

        Return 4 alternative queries that rephrase or approach the same question from different angles along with the original user query. So in total you have to return 5 queries - the original one and 4 variations. 
        """
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = query_llm.invoke(prompt)
                query_variations = response.queries  

                self.logger.info("Generated Query Variations:")
                for i, variation in enumerate(query_variations, 1):
                    self.logger.info(f"{i}. {variation}")

                return query_variations 
            
            except RateLimitError as e:
                wait = 60  
                self.logger.warning(f"⚠️ OpenAI rate limit hit. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)

    
    
    def retrieve_chunks_for_multiple_queries(self, queries: List[str], retriever): 
        all_retrieval_chunks = [] 

        for i, query in enumerate(queries, 1): 
            self.logger.info(f"Retrieving relevant chunks for query {i}: {query}")
            chunks = retriever.invoke(query) 
            all_retrieval_chunks.append(chunks)     
            
            # for j, doc in enumerate(chunks, 1): 
            #     self.logger.info(f"Document {j}:") 
            #     self.logger.info(f"{doc.page_content[:200]}...\n") 
            #     self.logger.info(f"Metadata: {doc.metadata}\n")
        
        self.logger.info("Multi-Query Retrieval Complete!")

        return all_retrieval_chunks
    


    def reciprocal_rank_fusion(self, chunk_lists: List[List[Document]], k: int = 60, filepath: str = os.path.join(PROJECT_ROOT, "debugging/retrieval_stage/reciprocal_rank_fusion_results.json")): 

        rrf_scores = defaultdict(float)  
        all_unique_chunks ={} 
        chunk_id_map ={} 
        chunk_counter =1 

        for i, query_chunks in enumerate(chunk_lists, 1):
            self.logger.info(f"Processing chunks for query {i} with {len(query_chunks)} retrieved documents.")
            
            for position, chunk in enumerate(query_chunks, 1): 
                chunk_content = chunk.page_content 
                chunk_metadata = chunk.metadata 

                if chunk_content not in chunk_id_map: 
                    chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                    all_unique_chunks[chunk_content] = chunk 
                    chunk_counter+=1 

                chunk_id = chunk_id_map[chunk_content]
                position_score = 1 / (k + position)

                rrf_scores[chunk_content] += position_score

                self.logger.info(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                self.logger.info(f"  Preview: {chunk_content[:80]}...")


        sorted_chunks = sorted(
            [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()], 
            key = lambda x: x[1], 
            reverse=True
        )

        rrf_chunks = [chunk for chunk, score in sorted_chunks]

        self.logger.info(f"✅ RRF Complete! Selected {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
 
        if self.log:
            with open(filepath, 'w', encoding='utf-8') as f:
                chunk_list = []
                for chunk, score in sorted_chunks:
                    chunk_data = {
                        "chunk_id": chunk_id_map[chunk.page_content],
                        "enhanced_content": chunk.page_content,
                        "metadata": {
                            "original_content": json.loads(chunk.metadata.get("original_content", "{}")),
                            "source": chunk.metadata.get("source", "unknown"),
                            "rrf_score": score
                        } 
                    } 
                    chunk_list.append(chunk_data)
                json.dump(chunk_list, f, indent=2, ensure_ascii=False)

        return rrf_chunks


        
    def create_retriever(self, retriever_type: str = "hybrid", vector_retriever_type: str = "similarity", all_chunks_path = os.path.join(PROJECT_ROOT, "vectorstore/chroma_db_production/all_chunks.json")): 
        self.logger.info(f"Creating a {retriever_type} retriever with vector retriever type: {vector_retriever_type}...")
        
        def get_vector_retriever(vector_retriever_type: str):
            if vector_retriever_type == "similarity":
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 9})  
            
            elif vector_retriever_type == "similarity_score_threshold": 
                retriever = self.vectorstore.as_retriever(
                                search_type="similarity_score_threshold",
                                search_kwargs={
                                    "k": 9,
                                    "score_threshold": 0.3
                                }
                            )
                
            elif vector_retriever_type == "mmr": 
                retriever = self.vectorstore.as_retriever(
                                search_type="mmr",
                                search_kwargs={
                                    "k": 9,           # Final number of docs
                                    "fetch_k": 10,    # Initial pool to select from
                                    "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
                                }
                            )
            return retriever
        
        def get_keyword_retriever(): 
            def list_to_docs(data):
                return [
                    Document(
                        page_content=d["page_content"],
                        metadata={                 
                            "original_content": json.dumps(d["metadata"].get("original_content", {})),
                            "source": d["metadata"].get("source", "unknown") 
                        }
                    )
                    for d in data
                ]
        
            with open(all_chunks_path, 'r', encoding='utf-8') as f:
                all_chunks= list_to_docs(json.load(f)) 

            keyword_retriever = BM25Retriever.from_documents(
                documents = all_chunks, 
            ) 
            keyword_retriever.k = 10 
            return keyword_retriever 
        
        def get_hybrid_retriever():
            vector_retriever = get_vector_retriever(vector_retriever_type = vector_retriever_type)
            keyword_retriever = get_keyword_retriever() 

            hybrid_retriever = EnsembleRetriever(
                retrievers = [vector_retriever, keyword_retriever], 
                weights = [0.7, 0.3]
            )
            return hybrid_retriever
        

        if retriever_type == "vector": 
            retriever = get_vector_retriever(vector_retriever_type = vector_retriever_type) 

        elif retriever_type == "hybrid": 
            retriever = get_hybrid_retriever()
                

        self.logger.info(f"✅ Created a {retriever_type} retriever with vector retriever type: {vector_retriever_type} successfully")

        return retriever
    


    def rerank_chunks(self, chunks: List[Document]): 
        import time
        from cohere.errors.too_many_requests_error import TooManyRequestsError

        if self.reranking_query is None:
            self.logger.warning("No reranking query available. Returning original chunk order") 
            return chunks 
        
        else:
            self.logger.info("Reranking chunks...") 
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    reranked_docs = self.reranker.compress_documents(query = self.reranking_query, documents = chunks) 

                    if self.log:
                        with open(os.path.join(PROJECT_ROOT, "debugging/retrieval_stage/reranked_chunks.json"), 'w', encoding='utf-8') as f:
                            chunk_list = []
                            for i, chunk in enumerate(reranked_docs, 1):
                                chunk_data = {
                                    "chunk_id": f"Reranked_Chunk_{i}",
                                    "enhanced_content": chunk.page_content,
                                    "metadata": {
                                        "original_content": json.loads(chunk.metadata.get("original_content", "{}")),
                                        "source": chunk.metadata.get("source", "unknown"),
                                        "reranking_query": self.reranking_query
                                    } 
                                } 
                                chunk_list.append(chunk_data)
                            json.dump(chunk_list, f, indent=2, ensure_ascii=False) 
                
                    self.logger.info("✅ Reranking complete! Exported reranked chunks to reranked_chunks.json")
                    return reranked_docs
                
                except TooManyRequestsError:
                    wait = 60 * (attempt + 1)  # 60s, 120s, 180s
                    self.logger.warning(f"Cohere rate limit hit. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)

            self.logger.warning("❌ Cohere reranker failed after all retries — returning unranked chunks.")
            return chunks
    


    def export_retrieved_chunks_to_json(self, chunks, retrieval_method: str, filename: str ="retrieved_chunks.json"):
        filepath = os.path.join(PROJECT_ROOT, "debugging/retrieval_stage", filename)

        export_data = []
        
        for i, doc in enumerate(chunks):
            chunk_data = {
                "chunk_id": i + 1,
                "enhanced_content": doc.page_content,
                "metadata": {
                    "original_content": json.loads(doc.metadata.get("original_content", "{}")),
                    "source": doc.metadata.get("source", "unknown"),
                    "retrieval_method": retrieval_method
                }
            }
            export_data.append(chunk_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Exported {len(export_data)} chunks to {filename}")
        return export_data 
    


    def run_simple_retrieval_pipeline(self, query: str, retriever_type: str = "hybrid", vector_retriever_type: str = "similarity_score_threshold"): 
        self.logger.info("🚀 Starting simple retrieval pipeline...")
        retriever = self.create_retriever(retriever_type= retriever_type, vector_retriever_type= vector_retriever_type)  

        self.logger.info(f"Retrieving relevant chunks for query: {query}")
        chunks = retriever.invoke(query) 

        self.export_retrieved_chunks_to_json(chunks = chunks, retrieval_method = retriever_type)  
        self.logger.info("✅ Simple retrieval pipeline completed successfully!")


    @traceable(name="RetrievalPipeline.run_retrieval_pipeline", run_type="retriever")
    def run_retrieval_pipeline(self, query: str, retriever_type: str = "hybrid", vector_retriever_type: str = "similarity_score_threshold"):  
        self.logger.info("🚀 Starting retrieval pipeline...")
        
        retriever = self.create_retriever(retriever_type= retriever_type, vector_retriever_type= vector_retriever_type)  
        query_variations = self.generate_query_variations(original_query = query)  
        all_retrieval_chunks = self.retrieve_chunks_for_multiple_queries(queries = query_variations, retriever = retriever) 
        rrf_chunks = self.reciprocal_rank_fusion(chunk_lists = all_retrieval_chunks) 
        reranked_chunks = self.rerank_chunks(chunks = rrf_chunks)

        self.logger.info("✅ Retrieval pipeline completed successfully!") 

        return reranked_chunks


        


if __name__ == "__main__": 
    ingestion_pipeline = IngestionPipeline(openai_api_key=os.getenv("OPENAI_API_KEY"), log=DEBUG) 
    vectorstore = ingestion_pipeline.run_ingestion_pipeline()
    retrieval_pipeline = RetrievalPipeline(openai_api_key=os.getenv("OPENAI_API_KEY"), cohere_api_key= os.getenv("CO_API_KEY"), vectorstore = vectorstore, log = DEBUG)

    query = "What problem does the Transformer architecture solve compared with recurrent sequence models?"
    retrieval_pipeline.run_retrieval_pipeline(query = query, retriever_type = "hybrid", vector_retriever_type = "similarity_score_threshold")
    



    


