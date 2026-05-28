import logging
import re
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import trim_messages                                                       

from src.backend.state.state import State, RouteDecision, QueryRewrite, InputGuardrails, OutputGuardrails
from src.backend.config.logging_config import DEBUG                    
from src.frontend.utils.utils import format_chat_history               
from src.backend.rag.retrieval_pipeline import RetrievalPipeline       
from src.backend.rag.response_synthesis_pipeline import ResponseSynthesisPipeline
from src.backend.prompts.orchestrator_agent_prompts import orchestrator_agent_prompts
from src.backend.prompts.response_generation_agent_prompts import response_generation_agent_prompts
from src.backend.prompts.query_rewriting_agent_prompts import query_rewriting_agent_prompts
from src.backend.prompts.input_guardrail_prompts import input_guardrail_prompts
from src.backend.prompts.output_guardrail_prompts import output_guardrail_prompts
from src.backend.prompts.blocking_agent_prompts import blocking_agent_prompts 
from src.backend.prompts.clarify_agent_prompts import clarify_agent_prompts




class Nodes:
    def __init__(self, openai_api_key, cohere_api_key, eval_mode : bool = False):
        self.openai_api_key = openai_api_key
        self.cohere_api_key = cohere_api_key
        self.eval_mode = eval_mode

    
    def input_guardrails(self, state:State, config: RunnableConfig):
        user_query = state["messages"][-1].content 
        llm = config["configurable"]["llm"]
        input_guardrail_llm = llm.with_structured_output(InputGuardrails)

        system_prompt = input_guardrail_prompts["v4"]

        trimmed_history = trim_messages(
            state["messages"], 
            max_tokens = 1000, 
            token_counter = llm, 
            strategy = "last", 
            include_system = True
        )  

        print(trimmed_history)

        response = input_guardrail_llm.invoke([SystemMessage(content=system_prompt)] + trimmed_history) 
       
        node_logger = logging.getLogger("input_guardrails")
        node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)
        node_logger.info("User Query: %s | Input Guardrails Decision: %s | Blocking Reason: %s", user_query, response.blocked, response.reason)

        return {
            "input_guardrail_blocked": response.blocked,
            "input_guardrail_reason": response.reason
        } 
        
     

    @staticmethod
    def route_from_input_guardrails(state: State) -> str:
        return "blocking_agent" if state["input_guardrail_blocked"] == True else "orchestrator"
    


    def blocking_agent(self, state:State, config:RunnableConfig):
        llm = config["configurable"]["llm"] 

        system_prompt = blocking_agent_prompts["v3"] 
        
        if state["input_guardrail_blocked"]:
            human_prompt = f""" 
            ASSESSMENT ON HUMAN INPUT:
            REASON FOR BLOCKAGE: {state["input_guardrail_reason"]} 
            """
        elif state["output_guardrail_blocked"]:
            human_prompt = f""" 
            ASSESSMENT ON AI OUTPUT:
            REASON FOR BLOCKAGE: {state["output_guardrail_reason"]} 
            """

        response = llm.invoke([SystemMessage(content = system_prompt), HumanMessage(content = human_prompt)]) 

        return {"messages": [AIMessage(content=response.content)]}
    


    def clarify_agent(self, state: State, config: RunnableConfig):
        node_logger = logging.getLogger("clarify_agent")
        node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)

        llm = config["configurable"]["llm"]

        latest_question = state["messages"][-1].content

        system_prompt = clarify_agent_prompts["v1"]

        user_prompt = f"""
        Latest User Question:
        {latest_question}
        """

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        node_logger.info("Clarify agent triggered | Query: %s", latest_question)
        return {"messages": [AIMessage(content=response.content)]}



    def orchestrator(self, state: State, config: RunnableConfig):
        user_query = state["messages"][-1].content
        llm = config["configurable"]["llm"]
        router_llm = llm.with_structured_output(RouteDecision)

        system_prompt = orchestrator_agent_prompts["v3"]

        trimmed_history = trim_messages(
            state["messages"], 
            max_tokens = 2000, 
            token_counter = llm, 
            strategy = "last", 
            include_system = True
        ) 

        decision = router_llm.invoke([SystemMessage(content=system_prompt)] + trimmed_history)

        node_logger = logging.getLogger("orchestrator")
        node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)
        node_logger.info("User Query: %s | Route Decision: %s", user_query, decision.route)

        return {"route": decision.route}



    @staticmethod
    def route_from_orchestrator(state: State) -> str:
        route = state["route"]
        if route == "rag":
            return "query_rewriting_agent"
        elif route == "clarify":
            return "clarify_agent"
        else:
            return "response_generation_agent"



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
                node_logger.warning(f"⚠️ OpenAI rate limit hit. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
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



    def output_guardrails(self, state:State, config:RunnableConfig):
        node_logger = logging.getLogger("output_guardrails")
        node_logger.setLevel(logging.INFO if DEBUG else logging.WARNING)

        response = state["messages"][-1].content
        llm = config["configurable"]["llm"] 

        pii_patterns = {
            "SSN":         r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b(?:\d[ -]?){13,16}\b',
            "password":    r'(?i)(password|passwd|secret\s*key)\s*[:=]\s*\S+',
            "bank":        r'(?i)(account\s*number|routing\s*number)\s*[:=]?\s*\d{8,17}',
        }

        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, response):
                 node_logger.warning("Output guardrail triggered: PII leakage (%s)", pii_type) 
                 return {
                     "output_guardrail_blocked" : True,
                     "output_guardrail_reason": "pii_leakage"
                 } 
                
            
        output_guardrail_llm = llm.with_structured_output(OutputGuardrails)

        decision = output_guardrail_llm.invoke([
            SystemMessage(content=output_guardrail_prompts["v1"]),
            HumanMessage(content=f"Evaluate this response:\n\n{response}")
        ]) 

        node_logger.info("AI Response: %s | Output Guardrails Decision: %s | Blocking Reason: %s", response, decision.blocked, decision.reason)

        if decision.blocked:
            return {
                "output_guardrail_blocked" : decision.blocked,
                "output_guardrail_reason": decision.reason
            } 
        else:
            return {
            "output_guardrail_blocked": False,
            "output_guardrail_reason": None
        } 

    

    @staticmethod
    def route_from_output_guardrails(state: State) -> str:
        return "blocking_agent" if state["output_guardrail_blocked"] == True else "end"
    



    


        

        
            

            

        




