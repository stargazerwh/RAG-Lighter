from typing_extensions import override
from .vectorStore import VectorStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChromaVS(VectorStore):
    def __init__(self, persist_directory, collection_name, embeddings_model):
        super().__init__(persist_directory, embeddings_model)
        self.vector_store = Chroma(embedding_function=self.embeddings_model, persist_directory=persist_directory, collection_name=collection_name)
        self.persist_directory = persist_directory

    @override
    def ingest(self, file_extension, data_path):
        docs = self.load_docs(file_extension, data_path)
        all_splits = self.split_docs(docs)
        _ = self.add_index(all_splits)
        print('🎉 All documents ingested and indexed')

    @override
    def similarity_search(self, question, k=2):
        return self.vector_store.similarity_search(question, k=k)

    def split_docs(self, docs, chunk_size=2000, chunk_overlap=100):
        print('⏳ Splitting documents...\n')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(docs)
        print(f'✅ {len(all_splits)} document splits created\n')
        return all_splits

        
    def add_index(self, all_splits):
        print('⏳ Adding documents to index...\n')
        _ = self.vector_store.add_documents(documents=all_splits)
        print('✅ Documents added to index\n')
        return _

    def load_docs(self, file_extension, data_path):
        print('⏳ Loading documents...\n')
        loader = DirectoryLoader(data_path, glob=file_extension)
        docs = loader.load()
        print(f'✅ {len(docs)} documents loaded\n')
        return docs
