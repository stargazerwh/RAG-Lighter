from llm import LLM
from rag import RAG

rag = RAG(
    embeddingsModel='all-MiniLM-L6-v2',
    dataPath='./data',
    fileExtension='**/*.pdf',
    persistDirectory='./chromaDb',
    collectionName='test'
)

llm = LLM(model='llama3', RAG=rag)

rag.ingestData()

llm.createGraph()

llm.test({"question": "Réponds en francais et focus toi uniquement sur la question qui suit ! Ne réponds rien qui changera de ca Comment gérer mon alimentation pendant un marathon ?"})