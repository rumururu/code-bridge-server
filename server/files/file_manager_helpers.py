"""Pure helper functions used by FileManager."""

from __future__ import annotations

from pathlib import Path


def validate_project_relative_path(base_path: Path, path: str) -> Path | None:
    """Resolve and validate a path within project base path."""
    try:
        normalized = path[1:] if path.startswith("/") else path
        full_path = (base_path / normalized).resolve()
        full_path.relative_to(base_path)
        return full_path
    except (ValueError, OSError):
        return None


def is_excluded_path(path: Path, excluded_patterns: set[str]) -> bool:
    """Return whether path basename matches one of excluded patterns."""
    name = path.name
    for pattern in excluded_patterns:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True
    return False


def fuzzy_match(query: str, target: str) -> bool:
    """Check if query fuzzy-matches target by ordered character inclusion."""
    query_idx = 0
    for char in target:
        if query_idx < len(query) and char == query[query_idx]:
            query_idx += 1
    return query_idx == len(query)


def match_score(query: str, target: str) -> float:
    """Calculate fuzzy match score (higher = better match)."""
    if query == target:
        return 1000.0
    if target.startswith(query):
        return 500.0 + (len(query) / len(target)) * 100
    if query in target:
        return 200.0 + (len(query) / len(target)) * 100

    score = 0.0
    query_idx = 0
    consecutive = 0
    for i, char in enumerate(target):
        if query_idx < len(query) and char == query[query_idx]:
            query_idx += 1
            consecutive += 1
            score += consecutive * 10
            if i == 0:
                score += 50
        else:
            consecutive = 0
    return score
