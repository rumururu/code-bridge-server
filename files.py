"""File management for project file browsing."""

import os
from pathlib import Path
from typing import Optional

# Sensitive file patterns to exclude
EXCLUDED_PATTERNS = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials.json",
    "secrets.json",
    "*.pem",
    "*.key",
    "*.p12",
    "*.keystore",
    "*.jks",
    ".git",
    ".svn",
    "node_modules",
    "__pycache__",
    ".dart_tool",
    ".idea",
    ".vscode",
    "build",
    ".gradle",
    "*.lock",
    "pubspec.lock",
    "package-lock.json",
    "yarn.lock",
}

# Maximum file size for content reading (1MB)
MAX_FILE_SIZE = 1 * 1024 * 1024

# Language detection by extension
LANGUAGE_MAP = {
    ".dart": "dart",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".txt": "text",
    ".xml": "xml",
    ".gradle": "groovy",
    ".kt": "kotlin",
    ".swift": "swift",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
}


class FileManager:
    """Manages file browsing with security validation."""

    def __init__(self, base_path: str):
        """Initialize with project base path."""
        self.base_path = Path(base_path).resolve()

    def _validate_path(self, path: str) -> Optional[Path]:
        """Validate path is within project and not excluded.

        Returns resolved path if valid, None otherwise.
        """
        # Normalize and resolve path
        try:
            if path.startswith("/"):
                path = path[1:]

            full_path = (self.base_path / path).resolve()

            # Security: Check path traversal
            full_path.relative_to(self.base_path)

            return full_path
        except (ValueError, OSError):
            return None

    def _is_excluded(self, path: Path) -> bool:
        """Check if path should be excluded."""
        name = path.name

        for pattern in EXCLUDED_PATTERNS:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True

        return False

    def list_directory(self, path: str = "") -> dict:
        """List directory contents.

        Args:
            path: Relative path within project (empty for root)

        Returns:
            Dictionary with entries list or error
        """
        full_path = self._validate_path(path)

        if full_path is None:
            return {"error": "Invalid path", "code": 400}

        if not full_path.exists():
            return {"error": "Path not found", "code": 404}

        if not full_path.is_dir():
            return {"error": "Not a directory", "code": 400}

        entries = []

        try:
            for item in sorted(full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if self._is_excluded(item):
                    continue

                entry = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                }

                if item.is_file():
                    try:
                        entry["size"] = item.stat().st_size
                    except OSError:
                        entry["size"] = 0

                entries.append(entry)

        except PermissionError:
            return {"error": "Permission denied", "code": 403}

        return {"entries": entries, "path": path or "/"}

    def read_file(self, path: str) -> dict:
        """Read file content.

        Args:
            path: Relative path within project

        Returns:
            Dictionary with content and language, or error
        """
        full_path = self._validate_path(path)

        if full_path is None:
            return {"error": "Invalid path", "code": 400}

        if not full_path.exists():
            return {"error": "File not found", "code": 404}

        if not full_path.is_file():
            return {"error": "Not a file", "code": 400}

        if self._is_excluded(full_path):
            return {"error": "Access denied", "code": 403}

        # Check file size
        try:
            size = full_path.stat().st_size
            if size > MAX_FILE_SIZE:
                return {
                    "error": f"File too large ({size} bytes). Max: {MAX_FILE_SIZE} bytes",
                    "code": 413,
                }
        except OSError:
            return {"error": "Cannot read file", "code": 500}

        # Determine language
        suffix = full_path.suffix.lower()
        language = LANGUAGE_MAP.get(suffix, "text")

        # Read content
        try:
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"error": "Binary file cannot be displayed", "code": 415}
        except PermissionError:
            return {"error": "Permission denied", "code": 403}
        except OSError as e:
            return {"error": f"Cannot read file: {e}", "code": 500}

        return {
            "content": content,
            "language": language,
            "path": path,
            "size": size,
        }

    def write_file(self, path: str, content: str, create_dirs: bool = False) -> dict:
        """Write content to a file.

        Args:
            path: Relative path within project
            content: File content to write
            create_dirs: Create parent directories if they don't exist

        Returns:
            Dictionary with success status or error
        """
        full_path = self._validate_path(path)

        if full_path is None:
            return {"error": "Invalid path", "code": 400}

        # Security: Don't allow writing to excluded patterns
        if self._is_excluded(full_path):
            return {"error": "Cannot write to this file type", "code": 403}

        # Check if parent directory exists
        parent_dir = full_path.parent
        if not parent_dir.exists():
            if create_dirs:
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    return {"error": f"Cannot create directory: {e}", "code": 500}
            else:
                return {"error": "Parent directory does not exist", "code": 404}

        # Check content size
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > MAX_FILE_SIZE:
            return {
                "error": f"Content too large ({len(content_bytes)} bytes). Max: {MAX_FILE_SIZE} bytes",
                "code": 413,
            }

        # Write file
        try:
            full_path.write_text(content, encoding="utf-8")
        except PermissionError:
            return {"error": "Permission denied", "code": 403}
        except OSError as e:
            return {"error": f"Cannot write file: {e}", "code": 500}

        return {
            "success": True,
            "path": path,
            "size": len(content_bytes),
        }

    def delete_file(self, path: str) -> dict:
        """Delete a file.

        Args:
            path: Relative path within project

        Returns:
            Dictionary with success status or error
        """
        full_path = self._validate_path(path)

        if full_path is None:
            return {"error": "Invalid path", "code": 400}

        if not full_path.exists():
            return {"error": "File not found", "code": 404}

        if not full_path.is_file():
            return {"error": "Not a file (use delete_directory for directories)", "code": 400}

        # Security: Don't allow deleting excluded patterns
        if self._is_excluded(full_path):
            return {"error": "Cannot delete this file type", "code": 403}

        try:
            full_path.unlink()
        except PermissionError:
            return {"error": "Permission denied", "code": 403}
        except OSError as e:
            return {"error": f"Cannot delete file: {e}", "code": 500}

        return {"success": True, "path": path}

    def search_files(self, query: str, limit: int = 50) -> dict:
        """Search for files matching query.

        Args:
            query: Search query (fuzzy match on filename)
            limit: Maximum number of results

        Returns:
            Dictionary with matching files list
        """
        if not query or len(query) < 1:
            return {"files": [], "query": query}

        query_lower = query.lower()
        matches = []

        def search_dir(dir_path: Path, relative_path: str = ""):
            """Recursively search directory."""
            try:
                for item in dir_path.iterdir():
                    if self._is_excluded(item):
                        continue

                    item_relative = f"{relative_path}/{item.name}" if relative_path else item.name

                    if item.is_dir():
                        search_dir(item, item_relative)
                    elif item.is_file():
                        name_lower = item.name.lower()
                        # Fuzzy match: check if query chars appear in order
                        if self._fuzzy_match(query_lower, name_lower):
                            score = self._match_score(query_lower, name_lower)
                            matches.append({
                                "path": item_relative,
                                "name": item.name,
                                "score": score,
                            })
            except PermissionError:
                pass

        search_dir(self.base_path)

        # Sort by score (higher is better) and limit
        matches.sort(key=lambda x: -x["score"])
        return {"files": matches[:limit], "query": query}

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Check if query fuzzy-matches target."""
        query_idx = 0
        for char in target:
            if query_idx < len(query) and char == query[query_idx]:
                query_idx += 1
        return query_idx == len(query)

    def _match_score(self, query: str, target: str) -> float:
        """Calculate match score (higher = better match)."""
        # Exact match
        if query == target:
            return 1000.0

        # Starts with query
        if target.startswith(query):
            return 500.0 + (len(query) / len(target)) * 100

        # Contains query as substring
        if query in target:
            return 200.0 + (len(query) / len(target)) * 100

        # Fuzzy match score based on consecutive matches
        score = 0.0
        query_idx = 0
        consecutive = 0
        for i, char in enumerate(target):
            if query_idx < len(query) and char == query[query_idx]:
                query_idx += 1
                consecutive += 1
                score += consecutive * 10  # Bonus for consecutive matches
                if i == 0:
                    score += 50  # Bonus for matching at start
            else:
                consecutive = 0

        return score

    def search_content(self, query: str, limit: int = 100, case_sensitive: bool = False) -> dict:
        """Search file contents for query.

        Args:
            query: Search query
            limit: Maximum number of results
            case_sensitive: Whether search is case-sensitive

        Returns:
            Dictionary with matching results including line numbers and context
        """
        if not query or len(query) < 2:
            return {"results": [], "query": query}

        results = []
        search_query = query if case_sensitive else query.lower()

        def search_file(file_path: Path, relative_path: str):
            """Search a single file for matches."""
            try:
                # Skip binary and large files
                size = file_path.stat().st_size
                if size > MAX_FILE_SIZE:
                    return

                content = file_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    search_line = line if case_sensitive else line.lower()
                    if search_query in search_line:
                        # Get context lines
                        context_before = lines[max(0, line_num - 2):line_num - 1]
                        context_after = lines[line_num:min(len(lines), line_num + 2)]

                        results.append({
                            "path": relative_path,
                            "line": line_num,
                            "content": line.strip(),
                            "context_before": [l.strip() for l in context_before],
                            "context_after": [l.strip() for l in context_after],
                        })

                        if len(results) >= limit:
                            return

            except (UnicodeDecodeError, PermissionError, OSError):
                pass

        def search_dir(dir_path: Path, relative_path: str = ""):
            """Recursively search directory."""
            if len(results) >= limit:
                return

            try:
                for item in dir_path.iterdir():
                    if len(results) >= limit:
                        return

                    if self._is_excluded(item):
                        continue

                    item_relative = f"{relative_path}/{item.name}" if relative_path else item.name

                    if item.is_dir():
                        search_dir(item, item_relative)
                    elif item.is_file():
                        search_file(item, item_relative)

            except PermissionError:
                pass

        search_dir(self.base_path)

        return {"results": results, "query": query, "total": len(results)}

    def create_directory(self, path: str) -> dict:
        """Create a directory.

        Args:
            path: Relative path within project

        Returns:
            Dictionary with success status or error
        """
        full_path = self._validate_path(path)

        if full_path is None:
            return {"error": "Invalid path", "code": 400}

        if full_path.exists():
            return {"error": "Path already exists", "code": 409}

        try:
            full_path.mkdir(parents=True, exist_ok=False)
        except PermissionError:
            return {"error": "Permission denied", "code": 403}
        except OSError as e:
            return {"error": f"Cannot create directory: {e}", "code": 500}

        return {"success": True, "path": path}

    def rename_file(self, old_path: str, new_path: str) -> dict:
        """Rename or move a file/directory.

        Args:
            old_path: Current relative path
            new_path: New relative path

        Returns:
            Dictionary with success status or error
        """
        old_full = self._validate_path(old_path)
        new_full = self._validate_path(new_path)

        if old_full is None:
            return {"error": "Invalid source path", "code": 400}
        if new_full is None:
            return {"error": "Invalid destination path", "code": 400}

        if not old_full.exists():
            return {"error": "Source not found", "code": 404}

        if new_full.exists():
            return {"error": "Destination already exists", "code": 409}

        if self._is_excluded(old_full) or self._is_excluded(new_full):
            return {"error": "Cannot rename protected files", "code": 403}

        try:
            # Create parent directory if needed
            new_full.parent.mkdir(parents=True, exist_ok=True)
            old_full.rename(new_full)
        except PermissionError:
            return {"error": "Permission denied", "code": 403}
        except OSError as e:
            return {"error": f"Cannot rename: {e}", "code": 500}

        return {"success": True, "old_path": old_path, "new_path": new_path}

    def copy_file(self, source_path: str, dest_path: str) -> dict:
        """Copy a file.

        Args:
            source_path: Source relative path
            dest_path: Destination relative path

        Returns:
            Dictionary with success status or error
        """
        import shutil

        source_full = self._validate_path(source_path)
        dest_full = self._validate_path(dest_path)

        if source_full is None:
            return {"error": "Invalid source path", "code": 400}
        if dest_full is None:
            return {"error": "Invalid destination path", "code": 400}

        if not source_full.exists():
            return {"error": "Source not found", "code": 404}

        if dest_full.exists():
            return {"error": "Destination already exists", "code": 409}

        if self._is_excluded(source_full) or self._is_excluded(dest_full):
            return {"error": "Cannot copy protected files", "code": 403}

        try:
            # Create parent directory if needed
            dest_full.parent.mkdir(parents=True, exist_ok=True)
            if source_full.is_dir():
                shutil.copytree(source_full, dest_full)
            else:
                shutil.copy2(source_full, dest_full)
        except PermissionError:
            return {"error": "Permission denied", "code": 403}
        except OSError as e:
            return {"error": f"Cannot copy: {e}", "code": 500}

        return {"success": True, "source": source_path, "dest": dest_path}

    def delete_directory(self, path: str) -> dict:
        """Delete a directory and all its contents.

        Args:
            path: Relative path within project

        Returns:
            Dictionary with success status or error
        """
        import shutil

        full_path = self._validate_path(path)

        if full_path is None:
            return {"error": "Invalid path", "code": 400}

        if not full_path.exists():
            return {"error": "Directory not found", "code": 404}

        if not full_path.is_dir():
            return {"error": "Not a directory", "code": 400}

        if self._is_excluded(full_path):
            return {"error": "Cannot delete protected directories", "code": 403}

        try:
            shutil.rmtree(full_path)
        except PermissionError:
            return {"error": "Permission denied", "code": 403}
        except OSError as e:
            return {"error": f"Cannot delete directory: {e}", "code": 500}

        return {"success": True, "path": path}


# Singleton per project
_file_managers: dict[str, FileManager] = {}


def get_file_manager(project_path: str) -> FileManager:
    """Get or create FileManager for project."""
    if project_path not in _file_managers:
        _file_managers[project_path] = FileManager(project_path)
    return _file_managers[project_path]
