import json
import os 
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from pydantic import BaseModel
from dotenv import load_dotenv  

from src.backend.rag.ingestion_pipeline import IngestionPipeline 
from src.backend.rag.retrieval_pipeline import RetrievalPipeline
from src.backend.config.logging_config import logging, DEBUG
from src.backend.llm.llm import get_llm, get_eval_llm
from src.backend.prompts.response_synthesis_prompts import response_synthesis_prompts



load_dotenv() 

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))


class ResponseSynthesisPipeline():
    
    def __init__(self, openai_api_key: str, log: bool = False, eval_mode: bool = False):
        self.log = log 
        self.logger = logging.getLogger(self.__class__.__name__) 
        self.logger.setLevel(logging.INFO if log else logging.WARNING)

        if eval_mode:
            self.logger.info("Initializing ResponseSynthesisPipeline in EVAL MODE.")
            self.llm = get_eval_llm(openai_key=openai_api_key)
        else:
            self.llm = get_llm(openai_key=openai_api_key)


    @traceable(name="ResponseSynthesisPipeline.generate_final_answer", run_type="chain")
    def generate_final_answer(self, chunks, query):   
        import time
        from openai import RateLimitError

        self.logger.info("🔍 Starting response synthesis...")    
        
        try: 
            system_prompt = response_synthesis_prompts["v2"]

            prompt_text = f"""
            User question:
            {query}

            Answer STRICTLY using the research paper excerpts below.
            Do NOT use any knowledge outside these excerpts.

            ================================================================
            DOCUMENT EXCERPTS:
            ================================================================
            """
        
            for i, chunk in enumerate(chunks):
                if "original_content" in chunk.metadata:   
                    original_data = json.loads(chunk.metadata["original_content"])

                    page_numbers = original_data.get("page_numbers")
                    if page_numbers:
                        if isinstance(page_numbers, list):
                            page_str = str(page_numbers[0]) if len(page_numbers) == 1 \
                                       else f"{min(page_numbers)}-{max(page_numbers)}"
                        else:
                            page_str = str(page_numbers)
                    else:
                        page_str = "Unknown"

                    source = chunk.metadata.get('source', 'Unknown')
                    prompt_text += f"\n--- Excerpt {i+1} | Source: {source} | Page: {page_str} ---\n"

                    raw_text = original_data.get("raw_text", "")
                    if raw_text:
                        prompt_text += f"TEXT:\n{raw_text}\n\n"
                    
                    tables_html = original_data.get("tables_html", [])
                    if tables_html:
                        prompt_text += "TABLES:\n"
                        for j, table in enumerate(tables_html):
                            prompt_text += f"Table {j+1}:\n{table}\n\n"
                            self.logger.info(f"📊 Added table to LLM prompt: Table {j+1} from Excerpt {i+1}")
                
                prompt_text += "\n"
            
            prompt_text += """
            ================================================================
            FINAL INSTRUCTIONS BEFORE ANSWERING:
            ================================================================
            1. Check: does the entity/concept in the question appear in the excerpts?
               → If NO  → reply: "I could not find this in the uploaded documents."
               → If YES → answer using ONLY the excerpt content below

            2. If no excerpts were provided at all:
               → reply: "I was not able to find any relevant information regarding this."

            3. Every sentence in your answer MUST end with a citation:
               (Source: <filename>, Page: <page number>)

            4. Never mix base model and big model results.
            5. Never use outside knowledge — even if you know the answer.

            FINAL ANSWER:
            """

            self.logger.info("📄 Constructed prompt for LLM")

            message_content = [{"type": "text", "text": prompt_text}]
            
            for chunk in chunks:
                if "original_content" in chunk.metadata:
                    original_data = json.loads(chunk.metadata["original_content"])
                    images_base64 = original_data.get("images_base64", [])
                    
                    for image_base64 in images_base64:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        })
                        self.logger.info(f"🖼️ Added image to LLM prompt: {image_base64[:10]}...")
           
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message_content)
            ]

            max_retries = 6
            for attempt in range(max_retries):
                try:
                    response = self.llm.invoke(messages)

                    if self.log:
                        filepath = os.path.join(PROJECT_ROOT, "debugging/response_synthesis_stage", "final_answer.json")
                        export_data = {
                            "user_query": query,
                            "final_answer": response.content 
                        }
                        with open(filepath, 'w', encoding='utf-8') as f: 
                            json.dump(export_data, f, indent=2, ensure_ascii=False)
                            
                    self.logger.info("✅ Response synthesis completed successfully.")
                    return response.content

                except RateLimitError:
                    wait = 10 ** attempt  
                    self.logger.warning(f"⚠️ Rate limit hit. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    
        except Exception as e:
            self.logger.error(f"❌ Answer generation failed: {e}")
            return "Sorry, I encountered an error while generating the answer." 



if __name__ == "__main__":
    ingestion_pipeline = IngestionPipeline(log=DEBUG, openai_api_key= os.getenv("OPENAI_API_KEY")) 
    vectorstore = ingestion_pipeline.run_ingestion_pipeline()
    
    query = "what is akshat's experience with full stack development?"

    retrieval_pipeline = RetrievalPipeline(openai_api_key= os.getenv("OPENAI_API_KEY"),cohere_api_key= os.getenv("CO_API_KEY"), vectorstore = vectorstore, log = DEBUG)
    final_chunks = retrieval_pipeline.run_retrieval_pipeline(query = query, retriever_type = "hybrid", vector_retriever_type = "similarity_score_threshold")

    response_synthesis_pipeline = ResponseSynthesisPipeline(openai_api_key= os.getenv("OPENAI_API_KEY"), log = DEBUG)
    final_answer = response_synthesis_pipeline.generate_final_answer(chunks = final_chunks, query = query)

    print("\n\n================= FINAL ANSWER =================\n") 
    print(final_answer)
