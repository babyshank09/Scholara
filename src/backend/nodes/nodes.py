import logging
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import trim_messages                                                       

from src.backend.state.state import State, RouteDecision, QueryRewrite 
from src.backend.config.logging_config import DEBUG                    
from src.frontend.utils.utils import format_chat_history               
from src.backend.rag.retrieval_pipeline import RetrievalPipeline       
from src.backend.rag.response_synthesis_pipeline import ResponseSynthesisPipeline
from src.backend.prompts.orchestrator_agent_prompts import orchestrator_agent_prompts
from src.backend.prompts.response_generation_agent_prompts import response_generation_agent_prompts
from src.backend.prompts.query_rewriting_agent_prompts import query_rewriting_agent_prompts




class Nodes:
    def __init__(self, openai_api_key, cohere_api_key, eval_mode : bool = False):
        self.openai_api_key = openai_api_key
        self.cohere_api_key = cohere_api_key
        self.eval_mode = eval_mode


    def orchestrator(self, state: State, config: RunnableConfig):
        user_query = state["messages"][-1].content
        llm = config["configurable"]["llm"]
        router_llm = llm.with_structured_output(RouteDecision)

        system_prompt = orchestrator_agent_prompts["v2"]

        decision = router_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ])

        node_logger = logging.getLogger("orchestrator")
        node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)
        node_logger.info("User Query: %s | Route Decision: %s", user_query, decision.route)

        return {"route": decision.route}


    @staticmethod
    def route_from_orchestrator(state: State) -> str:
        return "query_rewriting_agent" if state["route"] == "rag" else "response_generation_agent"


    def query_rewriting_agent(self, state: State, config: RunnableConfig):
        import time
        from openai import RateLimitError

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

        system_prompt = query_rewriting_agent_prompts["v2"]

        user_prompt = f"""
        Conversation History:
        {formatted_history}

        Latest User Question:
        {latest_question} 

        Current uploaded files:
        {config["configurable"]["current_embedded_files"]}

        Rewritten Standalone Query:
        """

        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = structured_llm.invoke([SystemMessage(content = system_prompt), HumanMessage(content = user_prompt)])
               
                node_logger = logging.getLogger("query_rewriting_agent")
                node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)
                node_logger.info("Latest Question: %s | Rewritten Query: %s", latest_question, result.standalone_query)

                return {"rewritten_query": result.standalone_query}
            
            except RateLimitError as e:
                wait = 60  
                self.logger.warning(f"⚠️ OpenAI rate limit hit. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)



    def retriever_agent(self, state: State, config: RunnableConfig):
        vectorstore = config["configurable"]["vectorstore"]
        retrieval_pipeline = RetrievalPipeline(openai_api_key= self.openai_api_key, cohere_api_key= self.cohere_api_key, vectorstore=vectorstore, log=DEBUG, eval_mode = self.eval_mode)
        retrieved_documents = retrieval_pipeline.run_retrieval_pipeline(
            query=state["rewritten_query"],
            retriever_type="hybrid",
            vector_retriever_type="similarity_score_threshold"             
        )                                                                                   
        return {"retrieved_docs": retrieved_documents}


    def response_generation_agent(self, state: State, config: RunnableConfig):
        llm = config["configurable"]["llm"]
        if state["route"] == "rag":
            response_synthesis_pipeline = ResponseSynthesisPipeline(openai_api_key=self.openai_api_key, log=DEBUG, eval_mode = self.eval_mode)
            final_answer = response_synthesis_pipeline.generate_final_answer(
                chunks=state["retrieved_docs"],
                query=state["rewritten_query"]
            )
            return {"messages": [AIMessage(content=final_answer)]}
        
        system_prompt = response_generation_agent_prompts["v2"]

        trimmed_history = trim_messages(
            state["messages"], 
            max_tokens = 4000, 
            token_counter = llm, 
            strategy = "last", 
            include_system = True
        ) 

        response = llm.invoke([SystemMessage(content=system_prompt)] + trimmed_history)
        return {"messages": [AIMessage(content=response.content)]}

