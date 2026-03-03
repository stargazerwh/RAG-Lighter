"""
Enhanced RAG with Parent-Child Chunking and Query Rewriting
增强型 RAG - 父子分块 + 查询改写

特性：
1. 父子分块：小块检索，大块生成
2. 查询策略：Direct / HyDE / Subquery / Auto
3. 配置驱动：统一 RAGConfig 配置
"""

import logging
from raglight.config.rag_config import RAGConfig
from raglight.config.settings import Settings
from raglight.rag.builder import Builder

# 设置日志
Settings.setup_logging()
logging.basicConfig(level=logging.INFO)


def example_basic():
    """基础用法 - 标准 RAG"""
    print("=" * 60)
    print("示例 1: 标准 RAG (无父子分块，无查询改写)")
    print("=" * 60)
    
    # 标准配置
    config = RAGConfig(
        llm="llama3.1:8b",
        provider=Settings.OLLAMA,
        k=5,
        # 禁用父子分块和查询改写
        use_parent_child_chunking=False,
        query_rewrite_strategy="Direct"
    )
    
    # 构建 RAG
    rag = (
        Builder()
        .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
        .with_vector_store(Settings.CHROMA, 
                          collection_name="standard_rag",
                          persist_directory="./db")
        .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
        .build_rag(config=config)
    )
    
    print("✓ 标准 RAG 构建完成")
    print(f"  - 父子分块: {config.use_parent_child_chunking}")
    print(f"  - 查询策略: {config.query_rewrite_strategy}")


def example_parent_child():
    """父子分块 RAG"""
    print("\n" + "=" * 60)
    print("示例 2: 父子分块 RAG")
    print("=" * 60)
    
    config = RAGConfig(
        llm="llama3.1:8b",
        provider=Settings.OLLAMA,
        k=3,
        # 启用父子分块
        use_parent_child_chunking=True,
        parent_chunk_size=2000,
        parent_chunk_overlap=200,
        child_chunk_size=400,
        child_chunk_overlap=50,
        # 标准查询
        query_rewrite_strategy="Direct"
    )
    
    rag = (
        Builder()
        .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
        .with_vector_store(Settings.CHROMA,
                          collection_name="pc_rag",
                          persist_directory="./pc_db")
        .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
        .build_rag(config=config)
    )
    
    print("✓ 父子分块 RAG 构建完成")
    print(f"  - 父块大小: {config.parent_chunk_size}")
    print(f"  - 子块大小: {config.child_chunk_size}")
    
    # 示例：处理文档
    # rag.vector_store.ingest("./documents")  # 会自动使用父子分块


def example_hyde():
    """HyDE 查询改写"""
    print("\n" + "=" * 60)
    print("示例 3: HyDE 查询改写 RAG")
    print("=" * 60)
    
    config = RAGConfig(
        llm="llama3.1:8b",
        provider=Settings.OLLAMA,
        k=3,
        use_parent_child_chunking=False,
        # 使用 HyDE 策略
        query_rewrite_strategy="HyDE"
    )
    
    rag = (
        Builder()
        .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
        .with_vector_store(Settings.CHROMA,
                          collection_name="hyde_rag",
                          persist_directory="./hyde_db")
        .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
        .build_rag(config=config)
    )
    
    print("✓ HyDE RAG 构建完成")
    print("  - 查询策略: HyDE")
    print("  - 流程: 用户问题 → 生成假设文档 → 两者都检索 → 生成答案")


def example_subquery():
    """子查询分解"""
    print("\n" + "=" * 60)
    print("示例 4: 子查询分解 RAG")
    print("=" * 60)
    
    config = RAGConfig(
        llm="llama3.1:8b",
        provider=Settings.OLLAMA,
        k=2,  # 每个子查询检索2个
        use_parent_child_chunking=False,
        # 使用子查询策略
        query_rewrite_strategy="Subquery"
    )
    
    rag = (
        Builder()
        .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
        .with_vector_store(Settings.CHROMA,
                          collection_name="subquery_rag",
                          persist_directory="./sub_db")
        .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
        .build_rag(config=config)
    )
    
    print("✓ Subquery RAG 构建完成")
    print("  - 查询策略: Subquery")
    print("  - 流程: 用户问题 → 分解为子查询 → 多查询检索 → 合并 → 生成")


def example_auto_strategy():
    """自动策略选择"""
    print("\n" + "=" * 60)
    print("示例 5: 自动策略选择 RAG (推荐)")
    print("=" * 60)
    
    config = RAGConfig(
        llm="llama3.1:8b",
        provider=Settings.OLLAMA,
        k=3,
        use_parent_child_chunking=True,  # 同时启用父子分块
        parent_chunk_size=2000,
        child_chunk_size=400,
        # 自动选择策略
        query_rewrite_strategy="Auto"
    )
    
    rag = (
        Builder()
        .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
        .with_vector_store(Settings.CHROMA,
                          collection_name="auto_rag",
                          persist_directory="./auto_db")
        .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
        .with_cross_encoder(Settings.HUGGINGFACE, 
                           model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        .build_rag(config=config)
    )
    
    print("✓ Auto Strategy RAG 构建完成")
    print("  - 父子分块: 启用")
    print("  - 查询策略: Auto (LLM 动态选择)")
    print("  - 重排序: 启用")
    print("\n  完整流程:")
    print("    用户问题 → LLM 选择策略 → 改写查询 → 子块检索 → 获取父块 → 重排序 → 生成答案")


def example_usage():
    """完整使用示例"""
    print("\n" + "=" * 60)
    print("示例 6: 完整使用流程")
    print("=" * 60)
    
    # 配置
    config = RAGConfig(
        llm="llama3.1:8b",
        provider=Settings.OLLAMA,
        system_prompt="你是一个有帮助的助手，基于提供的上下文回答问题。",
        k=3,
        use_parent_child_chunking=True,
        parent_chunk_size=2000,
        parent_chunk_overlap=200,
        child_chunk_size=400,
        child_chunk_overlap=50,
        query_rewrite_strategy="Auto"
    )
    
    # 构建
    rag = (
        Builder()
        .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
        .with_vector_store(Settings.CHROMA,
                          collection_name="my_app",
                          persist_directory="./my_db")
        .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
        .build_rag(config=config)
    )
    
    print("✓ RAG 系统构建完成")
    
    # 索引文档（自动使用父子分块）
    # rag.vector_store.ingest("./documents")
    
    # 查询（自动策略选择 + 查询改写）
    # response = rag.generate("你的复杂问题？")
    
    # 查看使用的策略
    # print(f"使用策略: {rag.get_last_strategy()}")
    # print(f"改写查询: {rag.get_last_queries()}")


def example_config_comparison():
    """配置对比"""
    print("\n" + "=" * 60)
    print("配置对比表")
    print("=" * 60)
    
    configs = [
        ("基础 RAG", RAGConfig(
            llm="llama3.1:8b",
            use_parent_child_chunking=False,
            query_rewrite_strategy="Direct"
        )),
        ("父子分块 RAG", RAGConfig(
            llm="llama3.1:8b",
            use_parent_child_chunking=True,
            parent_chunk_size=2000,
            child_chunk_size=400,
            query_rewrite_strategy="Direct"
        )),
        ("HyDE RAG", RAGConfig(
            llm="llama3.1:8b",
            use_parent_child_chunking=False,
            query_rewrite_strategy="HyDE"
        )),
        ("子查询 RAG", RAGConfig(
            llm="llama3.1:8b",
            use_parent_child_chunking=False,
            query_rewrite_strategy="Subquery"
        )),
        ("全自动 RAG (推荐)", RAGConfig(
            llm="llama3.1:8b",
            use_parent_child_chunking=True,
            parent_chunk_size=2000,
            child_chunk_size=400,
            query_rewrite_strategy="Auto"
        )),
    ]
    
    for name, cfg in configs:
        print(f"\n{name}:")
        print(f"  父子分块: {cfg.use_parent_child_chunking}")
        if cfg.use_parent_child_chunking:
            print(f"    - 父块: {cfg.parent_chunk_size}")
            print(f"    - 子块: {cfg.child_chunk_size}")
        print(f"  查询策略: {cfg.query_rewrite_strategy}")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       Enhanced RAG - 父子分块 + 查询改写示例               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        example_basic()
        example_parent_child()
        example_hyde()
        example_subquery()
        example_auto_strategy()
        example_usage()
        example_config_comparison()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
