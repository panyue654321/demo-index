"""DemoIndex package for PDF-to-markdown PageIndex-style trees."""

from __future__ import annotations

from typing import Any


def build_pageindex_tree(*args: Any, **kwargs: Any) -> dict:
    """Build a PageIndex-style tree from a PDF."""
    from .pipeline import build_pageindex_tree as _build_pageindex_tree

    return _build_pageindex_tree(*args, **kwargs)


def compare_tree(*args: Any, **kwargs: Any) -> dict:
    """Compare two PageIndex-style tree JSON files."""
    from .pipeline import compare_tree as _compare_tree

    return _compare_tree(*args, **kwargs)


def parse_query(*args: Any, **kwargs: Any):
    """Parse one retrieval query into a structured understanding object."""
    from .retrieval import parse_query as _parse_query

    return _parse_query(*args, **kwargs)


def retrieve_candidates(*args: Any, **kwargs: Any):
    """Run Stage 1 and Stage 2 retrieval and return a rich handoff object."""
    from .retrieval import retrieve_candidates as _retrieve_candidates

    return _retrieve_candidates(*args, **kwargs)


def localize_sections(*args: Any, **kwargs: Any):
    """Run Stage 3 tree localization over an existing Stage 1 + 2 result."""
    from .retrieval import localize_sections as _localize_sections

    return _localize_sections(*args, **kwargs)


def retrieve_tree_candidates(*args: Any, **kwargs: Any):
    """Run Stage 1 + Stage 2 + Stage 3 retrieval and return tree-localized sections."""
    from .retrieval import retrieve_tree_candidates as _retrieve_tree_candidates

    return _retrieve_tree_candidates(*args, **kwargs)


def expand_localized_sections(*args: Any, **kwargs: Any):
    """Run Stage 4 context expansion over an existing Stage 3 result."""
    from .retrieval import expand_localized_sections as _expand_localized_sections

    return _expand_localized_sections(*args, **kwargs)


def package_evidence(*args: Any, **kwargs: Any):
    """Run Stage 5 evidence packaging over an existing Stage 4 result."""
    from .retrieval import package_evidence as _package_evidence

    return _package_evidence(*args, **kwargs)


def retrieve_evidence(*args: Any, **kwargs: Any):
    """Run Stage 1 through Stage 5 retrieval and return packaged evidence."""
    from .retrieval import retrieve_evidence as _retrieve_evidence

    return _retrieve_evidence(*args, **kwargs)


__all__ = [
    "build_pageindex_tree",
    "compare_tree",
    "parse_query",
    "retrieve_candidates",
    "localize_sections",
    "retrieve_tree_candidates",
    "expand_localized_sections",
    "package_evidence",
    "retrieve_evidence",
]
