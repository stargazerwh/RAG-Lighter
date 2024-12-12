from ollama import Client
from dotenv import load_dotenv
from os import environ
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import json

from rag import RAG

load_dotenv()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class LLM() :
    def __init__(self, model, RAG, role='user'):
        self.model = model
        self.client = Client(
            host=environ.get('OLLAMA_HOST'),
            headers={'x-some-header': 'some-value'}
            )
        self.role = role
        self.rag = RAG
        self.graph = None

    def chat(self, input, k=2):
        response = self.client.chat(model=self.model, messages=[
            {
                'role': self.role,
                'content': input,
            },
        ])
        return response
    
    def retrieve(self, state, k=2):
        return self.rag.retrieve(state, k)
    
    def generate(self, state):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        promptJson = {
            "question": state["question"],
            "context": docs_content
        }
        prompt = json.dumps(promptJson)
        response = self.chat(prompt)
        return {"answer": response.message.content}
    
    def createGraph(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
    
    def test(self, state):
        response = self.graph.invoke(state)
        print(response["answer"])

    def testOllama(self):
        response = self.client.chat(model=self.model, messages=[
            {
                'role': self.role,
                'content': "Comment gérer mon alimentation pendant un marathon ?",
            },
        ])
        print(response.message.content)