from pydantic import BaseModel, HttpUrl, Field
from typing import List, Union, Literal

class FolderSource(BaseModel):
    name: str
    type: Literal["folder"] = "folder"
    path: str

class GitHubSource(BaseModel):
    name: str
    type: Literal["github"] = "github"
    urls: List[HttpUrl]

DataSource = Union[FolderSource, GitHubSource]
