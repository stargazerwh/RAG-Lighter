from pydantic import BaseModel, HttpUrl, Field
from typing import List, Union, Literal


class FolderSource(BaseModel):
    type: Literal["folder"] = "folder"
    path: str


class GitHubSource(BaseModel):
    type: Literal["github"] = "github"
    url: str


DataSource = Union[FolderSource, GitHubSource]
