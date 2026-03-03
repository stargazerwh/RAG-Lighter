"""
Parent-Child Chunking RAG Example
父子切块 RAG 使用示例

特点：
- 小块（child）用于精准向量检索
- 大块（parent）用于提供完整上下文给 LLM
"""

import logging
from raglight.config.settings import Settings
from raglight.embeddings import HuggingfaceEmbeddingsModel
from raglight.llm import OllamaModel
from raglight.vectorstore import ParentChildChromaVS
from raglight.document_processing import ParentChildProcessor
from raglight.rag import ParentChildRAG, ParentChildRAGBuilder

# 设置日志
Settings.setup_logging()
logging.basicConfig(level=logging.INFO)


def example_basic():
    """
    基础用法：手动组合组件
    """
    print("=" * 60)
    print("示例 1: 基础用法 - 手动组合组件")
    print("=" * 60)
    
    # 1. 创建嵌入模型
    print("\n[1/5] 初始化嵌入模型...")
    embeddings = HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2")
    
    # 2. 创建父子切块处理器
    print("[2/5] 创建父子切块处理器...")
    processor = ParentChildProcessor(
        parent_chunk_size=2000,   # 父块：2000 字符
        parent_chunk_overlap=200,
        child_chunk_size=400,     # 子块：400 字符（检索用）
        child_chunk_overlap=50
    )
    
    # 3. 创建父子切块向量存储
    print("[3/5] 创建向量存储...")
    vector_store = ParentChildChromaVS(
        children_collection_name="my_docs_children",
        parents_collection_name="my_docs_parents",
        embeddings_model=embeddings,
        persist_directory="./parent_child_db",
        k_children=10,   # 检索 10 个子块
        k_parents=3      # 返回 3 个父块
    )
    
    # 4. 创建 LLM
    print("[4/5] 创建 LLM...")
    llm = OllamaModel(
        model_name="llama3.1:8b",
        options={"num_ctx": 8192}
    )
    
    # 5. 创建父子切块 RAG
    print("[5/5] 创建 ParentChildRAG...")
    rag = ParentChildRAG(
        embedding_model=embeddings,
        vector_store=vector_store,
        llm=llm,
        k=3  # 返回 3 个父块
    )
    
    # 6. 处理文档并导入
    print("\n[6/6] 处理文档...")
    result = processor.process("./documents/article.txt")
    
    print(f"  - 生成父块: {len(result['parents'])} 个")
    print(f"  - 生成子块: {len(result['children'])} 个")
    
    # 导入到向量存储
    rag.ingest_parent_child(result)
    
    # 7. 查询
    print("\n[查询示例]")
    response = rag.generate("这篇文章的主要内容是什么？")
    print(f"回答: {response}")
    
    # 查看统计
    stats = rag.get_stats()
    print(f"\n[统计信息] {stats}")


def example_builder():
    """
    使用 Builder 模式快速创建（推荐）
    """
    print("\n" + "=" * 60)
    print("示例 2: 使用 Builder 模式（推荐）")
    print("=" * 60)
    
    # 使用 Builder 快速创建
    print("\n[1/3] 使用 Builder 构建 RAG...")
    builder = ParentChildRAGBuilder()
    
    rag, processor = (
        builder
        .with_embeddings(HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2"))
        .with_llm(OllamaModel(model_name="llama3.1:8b"))
        .with_vector_store(
            collection_name="my_collection",
            persist_directory="./parent_child_db",
            k_children=10,
            k_parents=3
        )
        .with_chunk_params(
            parent_size=2000,
            parent_overlap=200,
            child_size=400,
            child_overlap=50
        )
        .build_with_processor()  # 返回 RAG + Processor
    )
    
    print("  ✓ ParentChildRAG 创建成功")
    print(f"  ✓ ParentChildProcessor 创建成功")
    
    # 处理文档
    print("\n[2/3] 处理文档...")
    
    # 示例：直接处理文本
    sample_text = """
    RAG (Retrieval-Augmented Generation) 是一种将信息检索与文本生成结合的技术。
    它通过从外部知识库检索相关文档，然后将这些文档作为上下文提供给大语言模型，
    从而增强模型的回答质量和准确性。
    
    RAG 的主要优势包括：
    1. 减少幻觉：基于检索到的事实生成回答
    2. 知识更新：无需重新训练模型即可更新知识
    3. 可解释性：可以追溯回答来源
    
    父子切块（Parent-Child Chunking）是一种改进的 RAG 策略。
    小块用于精准检索，大块用于完整上下文，兼顾精准度和完整性。
    """
    
    result = processor.process_text(sample_text, source="demo")
    
    print(f"  - 父块数量: {len(result['parents'])}")
    print(f"  - 子块数量: {len(result['children'])}")
    print(f"  - 比例: 1:{len(result['children'])/len(result['parents']):.1f}")
    
    # 导入
    rag.ingest_parent_child(result)
    
    # 查询
    print("\n[3/3] 查询测试...")
    queries = [
        "什么是 RAG？",
        "父子切块有什么优势？",
        "RAG 如何减少幻觉？"
    ]
    
    for query in queries:
        print(f"\n  Q: {query}")
        response = rag.generate(query)
        print(f"  A: {response[:100]}...")
    
    # 统计
    stats = rag.get_stats()
    print(f"\n[最终统计]")
    print(f"  - 父块总数: {stats.get('parent_count', 0)}")
    print(f"  - 子块总数: {stats.get('child_count', 0)}")
    print(f"  - 平均子块/父块: {stats.get('ratio', 0):.1f}")


def example_file_processing():
    """
    批量处理文件示例
    """
    print("\n" + "=" * 60)
    print("示例 3: 批量处理文件")
    print("=" * 60)
    
    from pathlib import Path
    
    # 创建组件
    embeddings = HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2")
    
    processor = ParentChildProcessor(
        parent_chunk_size=1500,
        child_chunk_size=300
    )
    
    vector_store = ParentChildChromaVS(
        children_collection_name="file_docs",
        embeddings_model=embeddings,
        persist_directory="./file_db",
        k_children=15,
        k_parents=4
    )
    
    llm = OllamaModel(model_name="llama3.1:8b")
    
    rag = ParentChildRAG(
        embedding_model=embeddings,
        vector_store=vector_store,
        llm=llm,
        k=4
    )
    
    # 批量处理文件
    print("\n[批量处理文件]")
    docs_dir = Path("./documents")
    
    if not docs_dir.exists():
        print(f"  目录 {docs_dir} 不存在，创建示例文件...")
        docs_dir.mkdir(exist_ok=True)
        
        # 创建示例文档
        (docs_dir / "doc1.txt").write_text("""
        人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。
        它包括机器学习、自然语言处理、计算机视觉等多个子领域。
        """ * 10)
    
    total_parents = 0
    total_children = 0
    
    for file_path in docs_dir.glob("*.txt"):
        print(f"  处理: {file_path.name}")
        
        result = processor.process(str(file_path))
        rag.ingest_parent_child(result)
        
        total_parents += len(result['parents'])
        total_children += len(result['children'])
    
    print(f"\n  总计:")
    print(f"    - 文件数: {len(list(docs_dir.glob('*.txt')))}")
    print(f"    - 父块数: {total_parents}")
    print(f"    - 子块数: {total_children}")
    
    # 查询
    print("\n[查询]")
    response = rag.generate("人工智能包含哪些子领域？")
    print(f"回答: {response}")


def example_comparison():
    """
    对比：标准 RAG vs 父子切块 RAG
    """
    print("\n" + "=" * 60)
    print("示例 4: 标准 RAG vs 父子切块 RAG")
    print("=" * 60)
    
    # 长文本示例
    long_text = """
    机器学习是人工智能的核心技术之一。它使计算机能够从数据中学习模式，
    而无需明确编程。深度学习是机器学习的一个子集，使用多层神经网络
    来学习数据的复杂表示。
    
    神经网络受到生物神经元的启发，由相互连接的节点（神经元）组成。
    每个连接都有权重，通过学习算法调整这些权重来优化网络性能。
    反向传播是训练神经网络的核心算法，通过计算梯度来更新权重。
    
    Transformer 架构 revolutionized 自然语言处理领域。它使用自注意力机制
    来捕捉序列中的长距离依赖关系，而不需要递归结构。BERT、GPT 等
    预训练模型都基于 Transformer。
    """ * 20  # 重复以模拟长文档
    
    embeddings = HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2")
    llm = OllamaModel(model_name="llama3.1:8b")
    
    print("\n[标准 RAG - 单一切块大小]")
    from raglight.rag.builder import Builder
    from raglight.vectorstore import ChromaVS
    
    # 标准 RAG (chunk=500)
    standard_vs = ChromaVS(
        collection_name="standard_rag",
        embeddings_model=embeddings,
        persist_directory="./compare_db"
    )
    
    standard_rag = (
        Builder()
        .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
        .with_vector_store(Settings.CHROMA, 
                          collection_name="standard_rag",
                          persist_directory="./compare_db")
        .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
        .build_rag(k=3)
    )
    
    # 标准切块 (chunk_size=500)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([long_text])
    print(f"  切块数: {len(chunks)} (每个 500 字符)")
    
    # 标准 RAG 索引
    standard_rag.vector_store.ingest_texts([c.page_content for c in chunks])
    
    print("\n[父子切块 RAG]")
    pc_processor = ParentChildProcessor(
        parent_chunk_size=2000,  # 父块大
        child_chunk_size=400     # 子块小
    )
    
    result = pc_processor.process_text(long_text)
    print(f"  父块数: {len(result['parents'])} (每个 ~2000 字符)")
    print(f"  子块数: {len(result['children'])} (每个 ~400 字符)")
    
    pc_vs = ParentChildChromaVS(
        children_collection_name="pc_rag",
        embeddings_model=embeddings,
        persist_directory="./compare_db",
        k_children=10,
        k_parents=3
    )
    
    pc_rag = ParentChildRAG(
        embedding_model=embeddings,
        vector_store=pc_vs,
        llm=llm,
        k=3
    )
    
    pc_rag.ingest_parent_child(result)
    
    # 对比查询
    query = "反向传播算法是如何工作的？"
    print(f"\n[对比查询] '{query}'")
    
    print("\n  标准 RAG 检索到的上下文:")
    std_context = standard_rag.vector_store.similarity_search(query, k=3)
    for i, doc in enumerate(std_context):
        print(f"    [{i+1}] {len(doc.page_content)} 字符: {doc.page_content[:80]}...")
    
    print("\n  父子切块 RAG 检索到的上下文:")
    pc_context = pc_rag.vector_store.search(query, k=3)
    for i, doc in enumerate(pc_context):
        print(f"    [{i+1}] {len(doc.page_content)} 字符: {doc.page_content[:80]}...")
    
    print("\n  对比结论:")
    print("    - 标准 RAG: 检索精准，但上下文可能不完整")
    print("    - 父子 RAG: 检索精准 + 上下文完整")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       Parent-Child Chunking RAG 使用示例                  ║
    ║       父子切块 RAG - 小块检索，大块生成                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 运行示例
    try:
        example_builder()
        # example_basic()
        # example_file_processing()
        # example_comparison()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
