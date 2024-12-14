from llm import LLM
from rag import RAG

rag = RAG(
    embeddingsModel='all-MiniLM-L6-v2',
    dataPath='./data',
    fileExtension='**/*.pdf',
    persistDirectory='./chromaDb',
    collectionName='test'
)

llm = LLM(model='llama3', RAG=rag, systemFile='./systemPrompt.txt')

rag.ingestData()

llm.createGraph()

llm.test({"question": "Comment gérer mon alimentation pendant un marathon ?"})