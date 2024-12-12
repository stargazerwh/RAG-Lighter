from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class RAG():
    def __init__(self, embeddingsModel, dataPath, fileExtension, persistDirectory, collectionName):
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddingsModel)
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=persistDirectory, collection_name=collectionName)
        self.dataPath = dataPath
        self.fileExtension = fileExtension

    def loadDocs(self):
        print('Loading documents...\n')
        loader = DirectoryLoader(self.dataPath, glob=self.fileExtension)
        docs = loader.load()
        print(f'{len(docs)} documents loaded\n')
        return docs
    
    def splitDocs(self, docs, chunk_size=2000, chunk_overlap=100):
        print('Splitting documents...\n')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(docs)
        print(f'{len(all_splits)} document splits created\n')
        return all_splits
    
    def addIndex(self, all_splits):
        print('Adding documents to index...\n')
        _ = self.vector_store.add_documents(documents=all_splits)
        print('Documents added to index\n')
        return _
    
    def ingestData(self):
        docs = self.loadDocs()
        all_splits = self.splitDocs(docs)
        _ = self.addIndex(all_splits)
        print('All documents ingested and indexed')

    def retrieve(self, state, k=2):
        retrieved_docs = self.vector_store.similarity_search(state['question'], k=k)
        return {"context": retrieved_docs}
