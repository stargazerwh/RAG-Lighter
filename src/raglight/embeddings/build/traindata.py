"""
Build training data from PDF documents.
Generate QA pairs from document chunks using LLM.
"""

import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def load_corpus(files: List[str], chunk_size: int = 512, chunk_overlap: int = 50, verbose: bool = False) -> List[Document]:
    """
    Load PDF files and split into chunks using LangChain.
    
    Args:
        files: List of PDF file paths
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        verbose: Print progress information
        
    Returns:
        List of Document chunks
    """
    if verbose:
        print(f"Loading files {files}")
    
    all_docs = []
    for file_path in files:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
    
    if verbose:
        print(f"Loaded {len(all_docs)} pages from {len(files)} files")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    nodes = text_splitter.split_documents(all_docs)
    
    if verbose:
        print(f"Parsed {len(nodes)} nodes")
    
    return nodes


def documents_to_dataframe(documents: List[Document]) -> pd.DataFrame:
    """
    Convert LangChain Documents to pandas DataFrame.
    
    Args:
        documents: List of Document objects
        
    Returns:
        DataFrame with text and metadata
    """
    return pd.DataFrame({
        "text": [doc.page_content for doc in documents],
        "title": [doc.metadata.get("source", "unknown") + f"_{i}" for i, doc in enumerate(documents)],
        "page": [doc.metadata.get("page", -1) for doc in documents]
    })


def documents_to_qa(documents: List[Document], llm, verbose: bool = False) -> pd.DataFrame:
    """
    Convert documents to question-answer pairs using LLM.
    
    Args:
        documents: List of Document chunks
        llm: Language model for QA generation
        verbose: Print progress information
        
    Returns:
        DataFrame with question-answer pairs
    """
    prompt_template = """Please generate 3 question-answer pairs based on the following document fragment.
The questions should be answerable using only the information in the fragment.

Document Fragment:
{context}

Requirements:
1. Generate exactly 3 question-answer pairs
2. Questions should be clear and specific
3. Answers should be directly supported by the document fragment
4. Return result in JSON format: [{"question": "...", "answer": "..."}]

Response:"""
    
    all_qa = []
    
    for i, doc in enumerate(documents):
        if verbose:
            print(f"Generating QA for chunk {i+1}/{len(documents)}...")
        
        try:
            prompt = prompt_template.format(context=doc.page_content)
            response = llm.generate({"question": prompt})
            qa_pairs = _parse_qa_response(response)
            
            for qa in qa_pairs:
                qa["source"] = doc.metadata.get("source", "unknown")
                qa["page"] = doc.metadata.get("page", None)
            
            all_qa.extend(qa_pairs)
            
        except Exception as e:
            if verbose:
                print(f"Failed to generate QA for chunk {i+1}: {e}")
            continue
    
    if verbose:
        print(f"Generated total {len(all_qa)} QA pairs from {len(documents)} chunks")
    
    return pd.DataFrame(all_qa)


def _parse_qa_response(response: str) -> List[dict]:
    """Parse LLM response to extract QA pairs."""
    try:
        response = response.strip()
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        qa_pairs = json.loads(response)
        
        if not isinstance(qa_pairs, list):
            qa_pairs = [qa_pairs]
        
        validated = []
        for qa in qa_pairs:
            if "question" in qa and "answer" in qa:
                validated.append({
                    "question": qa["question"],
                    "answer": qa["answer"]
                })
        
        return validated
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return []


if __name__ == "__main__":
    from raglight.llm import OpenAIModel
    
    TRAIN_FILES = ["train.pdf"]
    VAL_FILES = ["val.pdf"]
    
    print("Loading training data...")
    train_nodes = load_corpus(TRAIN_FILES, chunk_size=512, verbose=True)
    
    print("\nLoading validation data...")
    val_nodes = load_corpus(VAL_FILES, chunk_size=512, verbose=True)
    
    train_df = documents_to_dataframe(train_nodes)
    val_df = documents_to_dataframe(val_nodes)
    
    print("\nInitializing LLM...")
    llm = OpenAIModel(model_name="gpt-4")
    
    print("\nGenerating QA pairs for training data...")
    train_qa_df = documents_to_qa(train_nodes, llm=llm, verbose=True)
    
    print("\nGenerating QA pairs for validation data...")
    val_qa_df = documents_to_qa(val_nodes, llm=llm, verbose=True)
    
    print(f"\nTrain QA: {len(train_qa_df)} rows")
    print(train_qa_df.head())
    
    print(f"\nVal QA: {len(val_qa_df)} rows")
    print(val_qa_df.head())
    
    train_qa_df.to_csv("train_qa.csv", index=False)
    val_qa_df.to_csv("val_qa.csv", index=False)
    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)
    
    print("\nSaved: train_qa.csv, val_qa.csv, train_data.csv, val_data.csv")
