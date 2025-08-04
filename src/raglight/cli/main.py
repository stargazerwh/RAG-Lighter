import typer
from pathlib import Path
import logging
import os

from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from typing_extensions import Annotated
from typing import Literal

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt as RichPrompt

import questionary
from quo.prompt import Prompt
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline

from .nltk_management import download_nltk_resources_if_needed

console = Console()

custom_style = questionary.Style([
    ("answer", "bold ansicyan"), 
])

def prompt_input():
    session = Prompt()
    return session.prompt(
        ">>> ", placeholder="<gray> enter your input here, type bye to quit</gray>"
    )

def print_llm_response(response: str):
    """Affiche la réponse LLM dans un panneau markdown cyan avec 🤖"""
    console.print(
        Panel(
            Markdown(response), border_style="cyan", title="[bold cyan]🤖[/bold cyan]"
        )
    )

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
        "telemetry",
        "langchain",
        "langchain_core",
        "langchain_core.tracing",
        "httpx",
        "urllib3",
        "requests",
        "chromadb",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)


def _run_interactive_chat_flow(chat_type: Literal["standard", "agentic"]):
    """
    Runs the entire interactive flow for setting up and starting a chat session.
    This function contains the shared logic for both standard and agentic RAG.

    Args:
        chat_type: Determines the type of pipeline to create ('standard' or 'agentic').
    """
    # download_nltk_resources_if_needed(console)
    console.print(
        "[bold magenta]👋 Welcome to the RAGLight Interactive Setup Wizard![/bold magenta]"
    )
    console.print(
        "[magenta]I will guide you through setting up your RAG pipeline.[/magenta]"
    )

    console.print("[bold blue]\n--- 📂 Step 1: Data Source ---[/bold blue]")
    data_path_str = RichPrompt.ask("[bold]Enter the path to the directory with your documents[/bold]")
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        console.print(
            f"[bold red]❌ Error: The path '{data_path_str}' is not a valid directory.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print("[bold blue]\n--- 💾 Step 2: Vector Database ---[/bold blue]")
    db_path = RichPrompt.ask(
        "[bold]Where should the vector database be stored?[/bold]",
        default=Settings.DEFAULT_PERSIST_DIRECTORY,
    )
    collection = RichPrompt.ask(
        "[bold]What is the name for the database collection?[/bold]",
        default=Settings.DEFAULT_COLLECTION_NAME,
    )

    console.print("[bold blue]\n--- 🧠 Step 3: Embeddings Model ---[/bold blue]")
    emb_provider = questionary.select(
        "Which embeddings provider do you want to use?",
        choices=[Settings.HUGGINGFACE, Settings.OLLAMA, Settings.OPENAI],
        default=Settings.HUGGINGFACE,
        style=custom_style
    ).ask()

    default_api_base = None
    if emb_provider == Settings.OLLAMA:
        default_api_base = Settings.DEFAULT_OLLAMA_CLIENT
    elif emb_provider == Settings.OPENAI:
        default_api_base = Settings.DEFAULT_OPENAI_CLIENT

    embeddings_base_url = RichPrompt.ask(
        "[bold]What is your base URL for the embeddings provider? (Not needed for HuggingFace)[/bold]",
        default=default_api_base,
    )
    emb_model = RichPrompt.ask(
        "[bold]Which embedding model do you want to use?[/bold]",
        default=Settings.DEFAULT_EMBEDDINGS_MODEL,
    )

    console.print("[bold blue]\n--- 🤖 Step 4: Language Model (LLM) ---[/bold blue]")
    llm_provider = questionary.select(
        "Which LLM provider do you want to use?",
        choices=[Settings.OLLAMA, Settings.MISTRAL, Settings.OPENAI, Settings.LMSTUDIO],
        default=Settings.OLLAMA,
        style=custom_style
    ).ask()
    
    llm_default_api_base = None
    if llm_provider == Settings.OLLAMA:
        llm_default_api_base = Settings.DEFAULT_OLLAMA_CLIENT
    elif llm_provider == Settings.OPENAI:
        llm_default_api_base = Settings.DEFAULT_OPENAI_CLIENT
    elif llm_provider == Settings.LMSTUDIO:
        llm_default_api_base = Settings.DEFAULT_LMSTUDIO_CLIENT

    llm_base_url = RichPrompt.ask(
        "[bold]What is your base URL for the LLM provider? (Not needed for Mistral)[/bold]",
        default=llm_default_api_base,
    )

    llm_model = RichPrompt.ask(
        "[bold]Which LLM do you want to use?[/bold]",
        default=Settings.DEFAULT_LLM,
    )
    k = questionary.select(
        "How many documents should be retrieved for context (k)?",
        choices=['5', '10', '15'],
        default=str(Settings.DEFAULT_K),
        style=custom_style
    ).ask()
    k = int(k)

    console.print("[bold green]\n✅ Configuration complete![/bold green]")

    try:
        console.print("[bold blue]\n--- ⏳ Step 5: Indexing Documents ---[/bold blue]")
        
        pipeline = None
        vector_store_for_indexing = None

        if chat_type == "standard":
            builder = Builder()
            builder.with_embeddings(emb_provider, model_name=emb_model, api_base=embeddings_base_url)
            builder.with_vector_store(
                Settings.CHROMA,
                persist_directory=db_path,
                collection_name=collection,
            )
            vector_store_for_indexing = builder.build_vector_store()
            pipeline = builder.with_llm(
                llm_provider,
                model_name=llm_model,
                api_base=llm_base_url,
                system_prompt=Settings.DEFAULT_SYSTEM_PROMPT,
            ).build_rag(k=k)
        
        elif chat_type == "agentic":
            vector_store_config = VectorStoreConfig(
                embedding_model=emb_model,
                database=Settings.CHROMA,
                persist_directory=db_path,
                provider=emb_provider,
                collection_name=collection,
            )
            agent_config = AgenticRAGConfig(
                provider=llm_provider,
                model=llm_model,
                k=k,
                system_prompt=Settings.DEFAULT_AGENT_PROMPT,
                max_steps=4,
                api_key=Settings.MISTRAL_API_KEY,
                api_base=llm_base_url,
            )
            pipeline = AgenticRAGPipeline(agent_config, vector_store_config)
            vector_store_for_indexing = pipeline.get_vector_store()

        should_index = True
        if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
            console.print(f"[yellow]A database seems to exist at '{db_path}'.[/yellow]")
            if not typer.confirm(
                "Do you want to re-index the data? (This will add documents to the existing collection)\nIf you don't want, existing database will be used.",
                default=False,
            ):
                should_index = False
        
        if should_index:
            vector_store_for_indexing.ingest(data_path=str(data_path))
            vector_store_for_indexing.ingest_code(repos_path=str(data_path))
            console.print("[bold green]✅ Indexing complete.[/bold green]")
        else:
            console.print("[bold yellow]Skipping indexing, using existing database.[/bold yellow]")

        console.print(
            "[bold blue]\n--- 💬 Step 6: Starting Chat Session ---[/bold blue]"
        )
        console.print(
            "[bold green]✅ RAG pipeline is ready. You can start chatting now![/bold green]"
        )
        console.print("[yellow]Type 'quit' or 'exit' to end the session.\n[/yellow]")

        while True:
            query = prompt_input()
            if query.lower() in ["bye", "exit", "quit"]:
                console.print("🤖 : See you soon 👋")
                break

            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold cyan]Waiting for response...[/bold cyan]\n"),
                transient=True,
                console=console,
            ) as progress:
                task = progress.add_task("", total=None)
                response = pipeline.generate(query)
                progress.update(task, completed=1)

            print_llm_response(response)

    except Exception as e:
        console.print(f"[bold red]❌ An unexpected error occurred: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(name="chat")
def interactive_chat_command():
    """
    Starts a guided, interactive session to configure and use a standard RAG pipeline.
    """
    _run_interactive_chat_flow(chat_type="standard")


@app.command(name="agentic-chat")
def interactive_agentic_chat_command():
    """
    Starts a guided, interactive session to configure and use an agentic RAG pipeline.
    """
    _run_interactive_chat_flow(chat_type="agentic")


@app.command(name="index", hidden=True)
def index_command(
    data_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Path to the directory containing documents to index.",
        ),
    ],
    db_path: Annotated[
        str, typer.Option("--db-path", "-db")
    ] = Settings.DEFAULT_PERSIST_DIRECTORY,
    collection: Annotated[
        str, typer.Option("--collection", "-c")
    ] = Settings.DEFAULT_COLLECTION_NAME,
    embeddings: Annotated[
        str, typer.Option("--embeddings-model", "-e")
    ] = Settings.DEFAULT_EMBEDDINGS_MODEL,
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
        console.print(
            f"[bold green]✅ Successfully indexed all documents from {data_path}[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]❌ An error occurred during indexing: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()