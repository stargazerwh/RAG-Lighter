# src/raglight/cli/main.py

import typer
from pathlib import Path
import logging
import os

from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from raglight.rag.rag import RAG
from typing_extensions import Annotated

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from quo.prompt import Prompt

console = Console()

def prompt_input():
    session = Prompt()
    return session.prompt(">>> ", placeholder="<gray> enter your input here, type bye to quit</gray>")


def print_llm_response(response: str):
    """Affiche la réponse LLM dans un panneau markdown cyan avec 🤖"""
    console.print(Panel(Markdown(response), border_style="cyan", title="[bold cyan]🤖[/bold cyan]"))

app = typer.Typer(
    help="RAGLight CLI: An interactive wizard to index and chat with your documents."
)

@app.callback()
def callback():
    """
    RAGLight CLI application.
    """
    Settings.setup_logging()
    for name in [
        "telemetry", "langchain", "langchain_core", "langchain_core.tracing",
        "httpx", "urllib3", "requests", "chromadb"
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)

@app.command(name="chat")
def interactive_chat_command():
    """
    Starts a guided, interactive session to configure, index, and chat with your data.
    """
    console.print("[bold magenta]👋 Welcome to the RAGLight Interactive Setup Wizard![/bold magenta]")
    console.print("[magenta]I will guide you through setting up your RAG pipeline.[/magenta]")
    
    console.print("[bold cyan]\n--- 📂 Step 1: Data Source ---[/bold cyan]")
    data_path_str = typer.prompt(
        "Enter the path to the directory with your documents"
    )
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        console.print(f"[bold red]❌ Error: The path '{data_path_str}' is not a valid directory.[/bold red]")
        raise typer.Exit(code=1)

    console.print("[bold cyan]\n--- 💾 Step 2: Vector Database ---[/bold cyan]")
    db_path = typer.prompt(
        "Where should the vector database be stored?",
        default=Settings.DEFAULT_PERSIST_DIRECTORY
    )
    collection = typer.prompt(
        "What is the name for the database collection?",
        default=Settings.DEFAULT_COLLECTION_NAME
    )
    
    console.print("[bold cyan]\n--- 🧠 Step 3: Embeddings Model ---[/bold cyan]")
    emb_provider = typer.prompt(
        f"Which embeddings provider do you want to use? ({Settings.HUGGINGFACE}, {Settings.OLLAMA}, ...)",
        default=Settings.HUGGINGFACE
    )
    emb_model = typer.prompt(
        "Which embedding model do you want to use?",
        default=Settings.DEFAULT_EMBEDDINGS_MODEL
    )
    
    console.print("[bold cyan]\n--- 🤖 Step 4: Language Model (LLM) ---[/bold cyan]")
    llm_provider = typer.prompt(
        f"Which LLM provider do you want to use? ({Settings.OLLAMA}, {Settings.MISTRAL}, ...)",
        default=Settings.OLLAMA
    )
    llm_model = typer.prompt(
        "Which LLM do you want to use? (e.g., llama3, gpt-4o)",
        default=Settings.DEFAULT_LLM
    )
    k = typer.prompt(
        "How many documents should be retrieved for context (k)?",
        type=int,
        default=5
    )
    
    console.print("[bold green]\n✅ Configuration complete![/bold green]")

    try:
        console.print("[bold cyan]\n--- ⏳ Step 5: Indexing Documents ---[/bold cyan]")
        
        should_index = True
        if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
            console.print(f"[yellow]A database seems to exist at '{db_path}'.[/yellow]")
            if not typer.confirm("Do you want to re-index the data? (This will add documents to the existing collection)\nIf you don't want, existing database will be used.", default=False):
                should_index = False

        builder = Builder()
        builder.with_embeddings(emb_provider, model_name=emb_model)
        builder.with_vector_store(
            Settings.CHROMA,
            persist_directory=db_path,
            collection_name=collection,
        )

        if should_index:
            vector_store = builder.build_vector_store()
            vector_store.ingest(data_path=str(data_path))
            vector_store.ingest_code(repos_path=str(data_path))
            console.print("[bold green]✅ Indexing complete.[/bold green]")
        else:
            console.print("[bold yellow]Skipping indexing, using existing database.[/bold yellow]")

        console.print("[bold cyan]\n--- 💬 Step 6: Starting Chat Session ---[/bold cyan]")
        
        rag_pipeline: RAG = builder.with_llm(
            llm_provider, 
            model_name=llm_model, 
            system_prompt=Settings.DEFAULT_SYSTEM_PROMPT
        ).build_rag(k=k)
        
        console.print("[bold green]✅ RAG pipeline is ready. You can start chatting now![/bold green]")
        console.print("[yellow]Type 'quit' or 'exit' to end the session.\n[/yellow]")
        
        while True:
            query = prompt_input()
            if query.lower() in ["bye", "exit", "quit"]:
                console.print("🤖 : See you soon 👋")
                break

            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold cyan]Waiting for response...[/bold cyan]"),
                transient=True,
                console=console,
            ) as progress:
                task = progress.add_task("", total=None)
                response = rag_pipeline.generate(query)
                progress.update(task, completed=1)

            print_llm_response(response)

    except Exception as e:
        console.print(f"[bold red]❌ An unexpected error occurred: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

@app.command(name="index", hidden=True)
def index_command(
    data_path: Annotated[Path, typer.Argument(
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True,
        help="Path to the directory containing documents to index."
    )],
    db_path: Annotated[str, typer.Option("--db-path", "-db")] = Settings.DEFAULT_PERSIST_DIRECTORY,
    collection: Annotated[str, typer.Option("--collection", "-c")] = Settings.DEFAULT_COLLECTION_NAME,
    embeddings: Annotated[str, typer.Option("--embeddings-model", "-e")] = Settings.DEFAULT_EMBEDDINGS_MODEL,
):
    """(Hidden) A direct command to index a directory for scripting purposes."""
    try:
        vector_store = (
            Builder()
            .with_embeddings(Settings.HUGGINGFACE, model_name=embeddings)
            .with_vector_store(
                Settings.CHROMA, persist_directory=db_path, collection_name=collection
            )
            .build_vector_store()
        )
        vector_store.ingest(data_path=str(data_path))
        vector_store.ingest_code(repos_path=str(data_path))
        console.print(f"[bold green]✅ Successfully indexed all documents from {data_path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ An error occurred during indexing: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
