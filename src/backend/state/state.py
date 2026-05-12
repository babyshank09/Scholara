from typing import Optional, List, Annotated, TypedDict, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain.schema import Document
from pydantic import BaseModel, Field


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
        description="Standalone version of the user's latest question"
    )