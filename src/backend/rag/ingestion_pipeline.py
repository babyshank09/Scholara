import json
import os 
import logging
from typing import List 

from unstructured.partition.pdf import partition_pdf 
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langsmith import traceable
from dotenv import load_dotenv  

from src.backend.config.logging_config import logging, DEBUG 
from src.backend.llm.llm import get_llm, get_eval_llm

load_dotenv() 

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))



class IngestionPipeline: 

    def __init__(self, openai_api_key: str, persist_directory: str = "vectorstore/chroma_db", log: bool = False, eval_mode: bool = False): 
        self.logger = logging.getLogger(self.__class__.__name__) 
        self.logger.setLevel(logging.INFO if log else logging.WARNING)

        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model = "text-embedding-3-small", openai_api_key = openai_api_key) 
        self.log = log
        
        debug_dirs = [
            "debugging/ingestion_stage",
            "debugging/retrieval_stage",
            "debugging/response_synthesis_stage",
        ]

        for path in debug_dirs:
            os.makedirs(os.path.join(PROJECT_ROOT, path), exist_ok=True)

        if eval_mode:
            self.logger.info("Initializing IngestionPipeline in EVAL MODE with detailed logging enabled.")
            self.llm = get_eval_llm(openai_key = openai_api_key)

        else:
            self.llm = get_llm(openai_key = openai_api_key) 

        




    def partition_document(self, file_path):
        self.logger.info("Partitioning document: %s", file_path)
        try:
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
            )
            self.logger.info("Extracted %d elements", len(elements))
            return elements
        
        except Exception:
            self.logger.exception("Failed to partition document: %s", file_path)
            raise   


    def create_chunks_by_title(self, elements):
        self.logger.info("🔨 Creating smart chunks...")
        
        try: 
            chunks = chunk_by_title(
                elements, 
                max_characters=3000, 
                new_after_n_chars=2400, 
                combine_text_under_n_chars=500 
            )
            self.logger.info(f"✅ Created {len(chunks)} chunks")
            return chunks
        
        except Exception:
            self.logger.exception("Failed to create chunks")
            raise   



    def seperate_content_types(self, chunk):  
        content_data = {
            'text' : chunk.text, 
            'tables' : [], 
            'images' : [], 
            'types' : ['text'],
            'page_numbers' : []
        }
        
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"): 
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__ 
                # print(element_type) 

                if hasattr(element.metadata, "page_number") and element.metadata.page_number:
                    if element.metadata.page_number not in content_data["page_numbers"]:
                        content_data["page_numbers"].append(element.metadata.page_number)

                if element_type == "Table": 
                    content_data["types"].append("table") 
                    table_html = getattr(element.metadata, "text_as_html", element.text) 
                    content_data["tables"].append(table_html) 

                if element_type == "Image": 
                    content_data["types"].append("image")
                    image_base64 = getattr(element.metadata, "image_base64", element.text) 
                    content_data["images"].append(image_base64) 

            
        
        content_data["types"] = list(set(content_data["types"]))
        return content_data  
    

    @traceable(name="IngestionPipeline.create_ai_enhanced_summary", run_type="llm")
    def create_ai_enhanced_summary(self, text: str, tables: List[str], images: List[str]) -> str: 
        try: 
            prompt_text = f"""You are creating a searchable description for document content retrieval.

            CONTENT TO ANALYZE:
            TEXT CONTENT:
            {text}

            """ 
            
            if tables: 
                prompt_text += "TABLES:\n"
                for i, table in enumerate(tables, 1):  
                    prompt_text += f"Table {i}:\n{table}\n\n"

            prompt_text += """
                    YOUR TASK:
                    Generate a comprehensive, searchable description that covers:

                    1. Key facts, numbers, and data points from text and tables
                    2. Main topics and concepts discussed  
                    3. Questions this content could answer
                    4. Visual content analysis (charts, diagrams, patterns in images)
                    5. Alternative search terms users might use

                    Make it detailed and searchable - prioritize findability over brevity.

                    SEARCHABLE DESCRIPTION:"""
            
            message_content = [{"type" : "text", "text": prompt_text}]  

            if images: 
                for image_base64 in images: 
                    message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
                    
            message = HumanMessage(content=message_content)
            response = self.llm.invoke([message])
            
            return response.content 
        
        except Exception as e: 
            self.logger.exception(f"     ❌ AI summary failed: {e}")
            raise
            


    def summarise_chunks(self, chunks, file_path): 
        langchain_documents = [] 
        total_chunks = len(chunks) 

        for i, chunk in enumerate(chunks, 1): 
            current_chunk = i 

            self.logger.info(f"Processing chunk {current_chunk}/{total_chunks}") 

            content_data = self.seperate_content_types(chunk = chunk) 
            
            self.logger.info(f"     Types found: {content_data['types']}")
            self.logger.info(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
            
            if content_data["tables"] or content_data["images"]: 
                try:
                    enhanced_content = self.create_ai_enhanced_summary(
                        content_data['text'],
                        content_data['tables'], 
                        content_data['images']
                    )
                    self.logger.info(f"     → AI summary created successfully")
                    self.logger.info(f"     → Enhanced content preview: {enhanced_content[:200]}...")
                except Exception as e:
                    self.logger.info(f"     ❌ AI summary failed: {e}")
                    enhanced_content = content_data['text']

            else:
                self.logger.info(f"     → Using raw text (no tables/images)")
                enhanced_content = content_data['text']


            doc = Document(
                page_content = f"The following chunk is from the source {file_path}:\n" + enhanced_content,
                metadata = {
                    "original_content" : json.dumps({
                        "raw_text": content_data['text'],
                        "tables_html": content_data['tables'],
                        "images_base64": content_data['images'],
                        "page_numbers": content_data["page_numbers"]
                    }),
                    "source" : file_path
                }
            ) 

            langchain_documents.append(doc) 

        self.logger.info(f"✅ Processed {len(langchain_documents)} chunks")
        return langchain_documents
    


    def export_chunks_to_json(self, chunks, filename="chunks_export.json"):
        filepath = os.path.join(PROJECT_ROOT, "debugging/ingestion_stage", filename)

        export_data = []
        
        for i, doc in enumerate(chunks):
            chunk_data = {
                "chunk_id": i + 1,
                "enhanced_content": doc.page_content,
                "metadata": {
                    "original_content": json.loads(doc.metadata.get("original_content", "{}")),
                    "source": doc.metadata.get("source", "unknown")
                }
            }
            export_data.append(chunk_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Exported {len(export_data)} chunks to {filename}")
        return export_data 
    

    
    def get_uploaded_filenames(self): 
        filenames = os.listdir(os.path.join(PROJECT_ROOT, "docs"))
        self.logger.info(f"Uploaded files in docs directory: {filenames}")
        return filenames
    


    def get_embedded_filenames(self, vectorstore): 
        existing = vectorstore.get(include=["metadatas"])
        embedded_filenames = list(set(metadata.get("source") for metadata in existing["metadatas"]))
        self.logger.info(f"Files already embedded in vectorstore: {embedded_filenames}")
        return embedded_filenames
    


    def get_processed_chunks(self, pdf_path: str): 
        source= os.path.basename(pdf_path)
        self.logger.info("Ingesting and creating chunks for %s", source)

        elements = self.partition_document(file_path = pdf_path)
        chunks = self.create_chunks_by_title(elements = elements) 
        processed_chunks = self.summarise_chunks(chunks = chunks, file_path = source)

        return processed_chunks
    


    def delete_chunks_by_source(self, vectorstore, source_filename: str, chunks_json_path: str = os.path.join(PROJECT_ROOT, "vectorstore/chroma_db_production/all_chunks.json")):
        def delete_chunks_from_list():
            if not os.path.exists(chunks_json_path):
                self.logger.warning("No chunks JSON found at %s", chunks_json_path)
                return 

            with open(chunks_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            original_count = len(data)
            filtered = [d for d in data if d["metadata"]["source"] != source_filename]
            removed_count = original_count - len(filtered)

            with open(chunks_json_path, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2, ensure_ascii=False)

            self.logger.info("Removed %d chunks for: %s from %s", removed_count, source_filename, os.path.basename(chunks_json_path))


        def delete_chunks_from_vectorstore():
            existing = vectorstore.get(include=["metadatas"])

            # source_filename = "attention-is-all-you-need.pdf"
            ids_to_delete = [
                    id_ for id_, md in zip(existing["ids"], existing["metadatas"])
                    if md.get("source") == source_filename
                ]
            
            vectorstore.delete(ids=ids_to_delete)
            self.logger.info(f"Deleted {len(ids_to_delete)} chunks from {source_filename}")

            current_embedded_filenames = self.get_embedded_filenames(vectorstore)
            self.logger.info(f"Current files embedded after deletion: {current_embedded_filenames}")  


        def delete_source_from_docs():
            docs_path = os.path.join(PROJECT_ROOT, "docs", source_filename) 
            if os.path.exists(docs_path): 
                os.remove(docs_path)
                self.logger.info(f"Deleted source file from docs: {source_filename}")
        

        if source_filename:
            delete_chunks_from_vectorstore() 
            delete_chunks_from_list() 
            delete_source_from_docs()
        
        else: 
            self.logger.warning("No source filename provided for deletion")

    
    @traceable(name="IngestionPipeline.run_ingestion_pipeline", run_type="chain")
    def run_ingestion_pipeline(self, persist_directory=os.path.join(PROJECT_ROOT, "vectorstore/chroma_db_production")):
        self.logger.info("🚀 Starting RAG Ingestion Pipeline...")

        os.makedirs(persist_directory, exist_ok=True)
        chunks_json_path = os.path.join(persist_directory, "all_chunks.json")

        def docs_to_list(docs):
            return [
                {
                    "page_content": doc.page_content,
                    "metadata": {                  
                        "original_content": json.loads(doc.metadata.get("original_content", "{}")),
                        "source": doc.metadata.get("source", "unknown")
                    }
                }
                for doc in docs
            ]

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

        def save_chunks(chunks):
            with open(chunks_json_path, "w", encoding="utf-8") as f:
                json.dump(docs_to_list(chunks), f, indent=2, ensure_ascii=False)
            self.logger.info("✅ Saved %d chunks to all_chunks.json", len(chunks))


        if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            self.logger.info("✅ Loaded existing vectorstore")

            if os.path.exists(chunks_json_path):
                with open(chunks_json_path, "r", encoding="utf-8") as f:
                    all_chunks = list_to_docs(json.load(f))
                self.logger.info("✅ Loaded existing chunks list (%d chunks)", len(all_chunks))
            else:
                all_chunks = []
                self.logger.warning("⚠️ No existing chunks JSON found, starting empty")

            uploaded_filenames = self.get_uploaded_filenames()
            embedded_filenames = self.get_embedded_filenames(vectorstore)

            newly_added = False
            for filename in uploaded_filenames:
                if filename not in embedded_filenames:                                     
                    self.logger.info("New file detected: %s - embedding...", filename)
                    new_pdf_path = os.path.join(PROJECT_ROOT, "docs", filename)
                    new_processed_chunks = self.get_processed_chunks(pdf_path=new_pdf_path) 
                    
                    if self.log:
                        self.export_chunks_to_json(chunks=new_processed_chunks, filename=f"{filename}_chunks.json")

                    vectorstore.add_documents(new_processed_chunks)
                    all_chunks.extend(new_processed_chunks)
                    newly_added = True
                    self.logger.info("✅ Added %d chunks from %s", len(new_processed_chunks), filename)
                else:
                    self.logger.info("File %s already embedded - skipping...", filename)

            if newly_added:
                save_chunks(all_chunks)

        else:
            uploaded_filenames = self.get_uploaded_filenames()
            final_processed_chunks = []

            for filename in uploaded_filenames:
                self.logger.info("File detected: %s - creating chunks...", filename)
                pdf_path = os.path.join(PROJECT_ROOT, "docs", filename)
                processed_chunks = self.get_processed_chunks(pdf_path=pdf_path) 

                if self.log:
                    self.export_chunks_to_json(chunks=processed_chunks, filename=f"{filename}_chunks.json")

                self.logger.info("✅ Created %d chunks from %s", len(processed_chunks), filename)
                final_processed_chunks.extend(processed_chunks)

            self.logger.info("--- Creating vector store ---")
            vectorstore = Chroma.from_documents(
                documents=final_processed_chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            self.logger.info("✅ Vector store created and saved to %s", persist_directory)

            save_chunks(final_processed_chunks)

        self.logger.info("🎉 Pipeline completed successfully!")
        return vectorstore





if __name__ == "__main__":
    pipeline  = IngestionPipeline(openai_api_key=os.getenv("OPENAI_API_KEY"), log=DEBUG) 
    db = pipeline.run_ingestion_pipeline()  

    # pipeline.delete_chunks_by_source(vectorstore = db, source_filename = "3459701_LST_Full Rim Design - Blade X-Y Stiffness.pdf") 
    # db = pipeline.run_ingestion_pipeline()

    








        
