import subprocess
import tempfile
from pathlib import Path
from typing import List
import logging
from ..models.data_source_model import GitHubSource


class GithubScrapper:
    """
    A utility class for cloning and managing GitHub repositories.

    Attributes:
        repo_urls (List[str]): A list of GitHub repository URLs to manage.
    """

    def __init__(self) -> None:
        """
        Initializes a GithubScrapper instance.

        Args:
            repositories (List[GitHubSource], optional): A list of GitHub repository sources to manage. Defaults to None.
        """
        pass

    @staticmethod
    def clone_github_repo(repo_url: str, clone_path: str, branch: str = "main") -> None:
        """
        Clones a GitHub repository to a specified local directory.

        Args:
            repo_url (str): The URL of the GitHub repository to clone.
            clone_path (str): The local directory where the repository will be cloned.
            branch (str, optional): The branch of the repository to clone. Defaults to "main".

        Raises:
            subprocess.CalledProcessError: If the git command fails.
            Exception: If an unexpected error occurs during cloning.
        """
        try:
            command = ["git", "clone", "--branch", branch, repo_url, clone_path]
            logging.info(f"Executing: {' '.join(command)}")

            subprocess.run(command, check=True)
            logging.info("✅ Clone successful!")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while cloning repository: {e}")
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")

    def clone_all(self, branch: str = "main") -> str:
        """
        Clones all repositories in the `repo_urls` list to a temporary directory.

        Args:
            branch (str, optional): The branch of each repository to clone. Defaults to "main".

        Returns:
            str: The path to the temporary directory containing the cloned repositories.
        """
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        logging.info(f"Cloning repositories into folder: {temp_path}")

        for url in self.repo_urls:
            repo_name = url.split("/")[-1].replace(".git", "")
            clone_path = temp_path / repo_name
            self.clone_github_repo(url, str(clone_path), branch)

        return str(temp_path)

    def get_urls(self) -> List[str]:
        """
        Retrieves the list of repository URLs managed by the GithubScrapper.

        Returns:
            List[str]: The list of repository URLs.
        """
        return self.repo_urls

    def set_repositories(self, repositories: List[str]) -> None:
        """
        Sets the list of GitHub repositories to manage.

        Args:
            repositories (List[GitHubSource]): A list of GitHub repository sources to manage.
        """
        self.repo_urls = repositories
