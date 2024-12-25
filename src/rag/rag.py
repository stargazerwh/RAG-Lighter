from ..vectorestore.vectorStore import VectorStore
from ..embeddings.embeddingsModel import EmbeddingsModel
from ..llm.llm import LLM
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAG():
    def __init__(self, embedding_model: EmbeddingsModel, vector_store: VectorStore, llm: LLM):
        self.embeddings = embedding_model.get_model()
        self.vector_store = vector_store
        self.llm = llm
        self.graph = self.createGraph()

    def retrieve(self, state, k=2):
        retrieved_docs = self.vector_store.similarity_search(state['question'], k=k)
        return {"context": retrieved_docs}
    
    def generate(self, state):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt_json = {
            "question": state["question"],
            "context": docs_content
        }
        response = self.llm.generate(prompt_json)
        return {"answer": response}
    
    def createGraph(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()

    def question_graph(self, question):
        state = {"question": question}
        response = self.graph.invoke(state)
        print(response["answer"])
        return response['answer']
