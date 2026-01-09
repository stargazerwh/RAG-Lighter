from typing import Any, Type, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class RetrieverInput(BaseModel):
    query: str = Field(description="The query to perform. Should be semantically close to target documents.")
    collection_name: Optional[str] = Field(default=None, description="Name of the collection to search in.")

class ClassRetrieverInput(BaseModel):
    query: str = Field(description="The name or description of the class to retrieve.")
    collection_name: Optional[str] = Field(default=None, description="Name of the collection to search in.")

class RetrieverTool(BaseTool):
    name: str = "retriever"
    description: str = "Uses semantic search to retrieve relevant parts of the code documentation or knowledge base."
    args_schema: Type[BaseModel] = RetrieverInput
    
    vector_store: Any = Field(exclude=True) 
    k: int = Field(exclude=True)

    def _run(self, query: str, collection_name: Optional[str] = None) -> str:
        retrieved_docs = self.vector_store.similarity_search(
            query, k=self.k, collection_name=collection_name
        )
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)]
        )

class ClassRetrieverTool(BaseTool):
    name: str = "class_retriever"
    description: str = "Retrieves class definitions and their locations in the codebase."
    args_schema: Type[BaseModel] = ClassRetrieverInput
    
    vector_store: Any = Field(exclude=True)
    k: int = Field(exclude=True)

    def _run(self, query: str, collection_name: Optional[str] = None) -> str:
        retrieved_classes = self.vector_store.similarity_search_class(
            query, k=self.k, collection_name=collection_name
        )
        return "\nRetrieved classes:\n" + "".join(
            [f"\n\n===== Class {i} =====\n{doc.page_content}\nSource File: {doc.metadata['source']}" 
             for i, doc in enumerate(retrieved_classes)]
        )