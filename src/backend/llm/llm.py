import streamlit as st 
from langchain_openai import ChatOpenAI 
from src.backend.config.llm_config import MODEL_NAME, TEMPERATURE 
import functools 


@st.cache_resource
def get_llm(openai_key):
    return ChatOpenAI(
        model= MODEL_NAME,
        api_key= openai_key,
        temperature= TEMPERATURE,
        stream_usage=True,
        max_retries=5,       
    ) 

@functools.lru_cache(maxsize=1)
def get_eval_llm(openai_key):
    return ChatOpenAI(
        model= MODEL_NAME,
        api_key= openai_key,
        temperature= TEMPERATURE,
        stream_usage=False, 
        max_retries=5,       
    ) 
