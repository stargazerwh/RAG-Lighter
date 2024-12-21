from src.rag.ragBuilder import RAGBuilder

rag = RAGBuilder() \
.with_embeddings('HuggingFace', model_name='all-MiniLM-L6-v2') \
.with_vector_store('Chroma', persist_directory='/Users/labess40/dev/rag-example/chromaDb', collection_name='test') \
.with_llm('Ollama', model_name='llama3', system_prompt_file='/Users/labess40/dev/rag-example/systemPrompt.txt') \
.build()

rag.question_graph("Comment gérer mon alimentation pendant un marathon ?")