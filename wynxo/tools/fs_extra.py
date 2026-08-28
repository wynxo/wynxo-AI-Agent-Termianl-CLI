"""Coding-agent naming aliases for the existing safe filesystem tools."""

from .files import ListDir, ReadFile
from .search import Glob, Grep


class ListDirectory(ListDir):
    name = "list_directory"
    description = "List project files as a bounded tree, excluding generated and VCS directories."


class FindFiles(Glob):
    name = "find_files"
    description = "Find files by glob pattern within the workspace."


class SearchText(Grep):
    name = "search_text"
    description = "Search workspace text with a regular expression and optional context."
