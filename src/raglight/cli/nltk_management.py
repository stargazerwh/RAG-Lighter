from rich.console import Console
import nltk


def download_nltk_resources_if_needed(console: Console):
    """Download necessary NLTK resources if they are not already available."""
    required_resources = ["punkt", "stopwords"]
    for resource in required_resources:
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource == "punkt"
                else f"corpora/{resource}"
            )
        except LookupError:
            console.print(
                f"[bold yellow]NLTK resource '{resource}' not found. Downloading...[/bold yellow]"
            )
            nltk.download(resource, quiet=True)
            console.print(
                f"[bold green]✅ Resource '{resource}' downloaded.[/bold green]"
            )
