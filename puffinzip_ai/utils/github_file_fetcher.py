# PuffinZipAI_Project/puffinzip_ai/utils/github_file_fetcher.py
"""
GitHub File Fetcher — downloads real-world text files from trusted GitHub
repositories for use as benchmark training data.

Design goals:
  * **Trusted sources only** — curated allowlist of high-star repos +
    optional star-count threshold for auto-discovered repos.
  * **Size-normalized** — files are filtered to a configurable byte range
    so all benchmark items are roughly the same size.
  * **Cached locally** — downloaded files are stored in a local cache
    directory to avoid repeated API calls (GitHub rate-limits unauthenticated
    requests to 60/hour).
  * **Graceful degradation** — network failures, rate limits, or missing
    repos are handled silently; the caller gets whatever files are available
    in the cache with an empty-list fallback.
  * **Text-only** — only files with whitelisted extensions are fetched;
    binary files and very large files are skipped.

Usage from the evolutionary optimizer:
    from puffinzip_ai.utils.github_file_fetcher import GitHubFileFetcher

    fetcher = GitHubFileFetcher()
    items = fetcher.get_benchmark_items(count=20)
    # items is a list[str] of file contents, each within the target size range
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
#  Configuration imports (with safe fallbacks)
# ---------------------------------------------------------------------------
_GITHUB_CACHE_DIR: str = ""
_GITHUB_TARGET_SIZE_MIN: int = 1024            # 1 KB
_GITHUB_TARGET_SIZE_MAX: int = 51200           # 50 KB
_GITHUB_API_TOKEN: str | None = None
_GITHUB_MIN_STARS: int = 500
_GITHUB_FILE_EXTENSIONS: list[str] = []
_GITHUB_TRUSTED_REPOS: list[str] = []
_GITHUB_FETCH_TIMEOUT: int = 15

_logger: logging.Logger = logging.getLogger("GitHubFileFetcher")

try:
    from ..config import (
        GITHUB_CACHE_DIR,
        GITHUB_TARGET_FILE_SIZE_MIN,
        GITHUB_TARGET_FILE_SIZE_MAX,
        GITHUB_API_TOKEN,
        GITHUB_MIN_STARS,
        GITHUB_FILE_EXTENSIONS,
        GITHUB_TRUSTED_REPOS,
        GITHUB_FETCH_TIMEOUT,
    )
    _GITHUB_CACHE_DIR = GITHUB_CACHE_DIR
    _GITHUB_TARGET_SIZE_MIN = GITHUB_TARGET_FILE_SIZE_MIN
    _GITHUB_TARGET_SIZE_MAX = GITHUB_TARGET_FILE_SIZE_MAX
    _GITHUB_API_TOKEN = GITHUB_API_TOKEN
    _GITHUB_MIN_STARS = GITHUB_MIN_STARS
    _GITHUB_FILE_EXTENSIONS = GITHUB_FILE_EXTENSIONS
    _GITHUB_TRUSTED_REPOS = GITHUB_TRUSTED_REPOS
    _GITHUB_FETCH_TIMEOUT = GITHUB_FETCH_TIMEOUT
except ImportError:
    pass

try:
    from ..logger import setup_logger
    _logger = setup_logger("GitHubFileFetcher", log_level=logging.INFO)
except ImportError:
    if not _logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        _logger.addHandler(_h)
        _logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
#  Curated allowlist of trusted repos (high-star, well-known orgs)
#  Each entry is "owner/repo".  Only text files from these repos are fetched.
# ---------------------------------------------------------------------------
DEFAULT_TRUSTED_REPOS: list[str] = [
    # Python ecosystem
    "python/cpython",
    "pallets/flask",
    "psf/requests",
    "django/django",
    "fastapi/fastapi",
    # JavaScript / TypeScript
    "nodejs/node",
    "expressjs/express",
    "microsoft/TypeScript",
    # Rust
    "rust-lang/rust",
    "denoland/deno",
    # Go
    "golang/go",
    # Documentation / prose
    "github/docs",
    "mdn/content",
    # Data / config heavy
    "home-assistant/core",
    "ansible/ansible",
    # C / systems
    "torvalds/linux",
    "git/git",
]

DEFAULT_FILE_EXTENSIONS: list[str] = [
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".txt", ".rst", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".xml",
    ".rs", ".go", ".java", ".c", ".h", ".cpp",
    ".rb", ".sh", ".bat", ".ps1", ".cfg", ".ini",
]


# ---------------------------------------------------------------------------
#  Cache metadata schema
# ---------------------------------------------------------------------------
# The cache directory structure:
#   data/github_cache/
#     _index.json           — maps hash → {repo, path, size, fetched_at}
#     <sha256_hex>.txt      — raw file contents
#

CACHE_INDEX_FILENAME = "_index.json"
MAX_CACHE_AGE_SECONDS = 7 * 24 * 3600  # Re-fetch after 7 days


class GitHubFileFetcher:
    """Fetches and caches real-world text files from trusted GitHub repos.

    Attributes:
        cache_dir:      Local directory for cached files.
        target_min:     Minimum file size in bytes.
        target_max:     Maximum file size in bytes.
        trusted_repos:  Allowlist of ``owner/repo`` strings.
        extensions:     File extension whitelist.
        api_token:      Optional GitHub personal access token (from env or config).
        min_stars:      Minimum star count for auto-discovered repos.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        target_min: int | None = None,
        target_max: int | None = None,
        trusted_repos: list[str] | None = None,
        extensions: list[str] | None = None,
        api_token: str | None = None,
        min_stars: int | None = None,
        logger_instance: logging.Logger | None = None,
    ):
        self.logger = logger_instance or _logger

        # --- Resolve cache directory ---
        self.cache_dir = cache_dir or _GITHUB_CACHE_DIR
        if not self.cache_dir:
            # Fallback: <project_root>/data/github_cache/
            try:
                from ..config import DATA_DIR
                self.cache_dir = os.path.join(DATA_DIR, "github_cache")
            except ImportError:
                self.cache_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "github_cache",
                )
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- Size constraints ---
        self.target_min = target_min if target_min is not None else _GITHUB_TARGET_SIZE_MIN
        self.target_max = target_max if target_max is not None else _GITHUB_TARGET_SIZE_MAX

        # --- Trust & filtering ---
        self.trusted_repos = trusted_repos or _GITHUB_TRUSTED_REPOS or list(DEFAULT_TRUSTED_REPOS)
        self.extensions = extensions or _GITHUB_FILE_EXTENSIONS or list(DEFAULT_FILE_EXTENSIONS)
        self.min_stars = min_stars if min_stars is not None else _GITHUB_MIN_STARS

        # --- API token (prefer env var → config → None) ---
        self.api_token = api_token or _GITHUB_API_TOKEN or os.environ.get("GITHUB_TOKEN")
        self.timeout = _GITHUB_FETCH_TIMEOUT

        # --- Load cache index ---
        self._index: dict[str, dict] = self._load_index()

        self.logger.info(
            f"GitHubFileFetcher initialised: cache={self.cache_dir}, "
            f"repos={len(self.trusted_repos)}, size={self.target_min}-{self.target_max}B, "
            f"token={'set' if self.api_token else 'none'}"
        )

    # ------------------------------------------------------------------
    #  Cache management
    # ------------------------------------------------------------------

    def _index_path(self) -> str:
        return os.path.join(self.cache_dir, CACHE_INDEX_FILENAME)

    def _load_index(self) -> dict[str, dict]:
        path = self._index_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                self.logger.warning("Corrupted cache index — rebuilding.")
        return {}

    def _save_index(self) -> None:
        try:
            with open(self._index_path(), "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache index: {e}")

    def _cache_key(self, repo: str, file_path: str) -> str:
        """Deterministic hash key for a repo+path pair."""
        raw = f"{repo}:{file_path}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _is_cached(self, key: str) -> bool:
        """Check if a file is cached and not stale."""
        if key not in self._index:
            return False
        entry = self._index[key]
        cache_file = os.path.join(self.cache_dir, f"{key}.txt")
        if not os.path.isfile(cache_file):
            return False
        fetched_at = entry.get("fetched_at", 0)
        if time.time() - fetched_at > MAX_CACHE_AGE_SECONDS:
            return False
        return True

    def _read_cached(self, key: str) -> str | None:
        """Read a cached file's content."""
        cache_file = os.path.join(self.cache_dir, f"{key}.txt")
        try:
            with open(cache_file, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None

    def _write_cache(self, key: str, content: str, repo: str, file_path: str) -> None:
        """Write content to cache and update index."""
        cache_file = os.path.join(self.cache_dir, f"{key}.txt")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content)
            self._index[key] = {
                "repo": repo,
                "path": file_path,
                "size": len(content),
                "fetched_at": time.time(),
            }
        except Exception as e:
            self.logger.warning(f"Cache write failed for {repo}:{file_path}: {e}")

    # ------------------------------------------------------------------
    #  GitHub API helpers
    # ------------------------------------------------------------------

    def _api_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PuffinZipAI-BenchmarkFetcher/1.0",
        }
        if self.api_token:
            headers["Authorization"] = f"token {self.api_token}"
        return headers

    def _check_rate_limit(self) -> bool:
        """Return True if we have remaining API calls."""
        try:
            import requests
            resp = requests.get(
                "https://api.github.com/rate_limit",
                headers=self._api_headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                remaining = data.get("resources", {}).get("core", {}).get("remaining", 0)
                if remaining < 5:
                    reset_at = data.get("resources", {}).get("core", {}).get("reset", 0)
                    self.logger.warning(
                        f"GitHub API rate limit nearly exhausted ({remaining} remaining). "
                        f"Resets at {time.ctime(reset_at)}. Using cached files only."
                    )
                    return False
                return True
        except Exception:
            pass
        return False

    def _verify_repo_trusted(self, repo: str) -> bool:
        """Check that a repo meets trust criteria (on allowlist OR has enough stars)."""
        if repo in self.trusted_repos:
            return True
        # Auto-discover: check star count
        try:
            import requests
            resp = requests.get(
                f"https://api.github.com/repos/{repo}",
                headers=self._api_headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                stars = resp.json().get("stargazers_count", 0)
                if stars >= self.min_stars:
                    self.logger.info(f"Auto-trusted {repo} ({stars} stars >= {self.min_stars})")
                    return True
                else:
                    self.logger.info(f"Rejected {repo}: only {stars} stars (need {self.min_stars})")
        except Exception:
            pass
        return False

    def _list_repo_files(self, repo: str, path: str = "",
                         max_files: int = 100) -> list[dict]:
        """List files in a repo directory via the GitHub Trees API.

        Uses the recursive tree endpoint for efficiency (single API call
        for the entire repo tree).

        Returns:
            List of dicts with keys: path, size, sha, type
        """
        try:
            import requests
            # Use the Git Trees API with recursive=1 for a single-call tree walk
            resp = requests.get(
                f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1",
                headers=self._api_headers(),
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                self.logger.warning(f"Tree API failed for {repo}: HTTP {resp.status_code}")
                return []

            tree = resp.json().get("tree", [])

            # Filter: blobs only, matching extensions, within size range
            matching = []
            ext_set = set(self.extensions)
            for item in tree:
                if item.get("type") != "blob":
                    continue
                fpath = item.get("path", "")
                fsize = item.get("size", 0)

                # Extension check
                _, ext = os.path.splitext(fpath)
                if ext.lower() not in ext_set:
                    continue

                # Size check
                if fsize < self.target_min or fsize > self.target_max:
                    continue

                # Skip vendored / generated / test fixture paths
                lower_path = fpath.lower()
                if any(skip in lower_path for skip in (
                    "vendor/", "node_modules/", "dist/", "build/",
                    ".min.", "generated", "fixture", "__pycache__",
                    "migrations/", "locale/", ".lock",
                )):
                    continue

                matching.append({
                    "path": fpath,
                    "size": fsize,
                    "sha": item.get("sha", ""),
                })

                if len(matching) >= max_files:
                    break

            return matching
        except Exception as e:
            self.logger.warning(f"Error listing files for {repo}: {e}")
            return []

    def _download_file(self, repo: str, file_path: str) -> str | None:
        """Download a single file's raw content from GitHub."""
        try:
            import requests
            # Use the raw content endpoint (no API rate cost for public repos
            # when accessed via raw.githubusercontent.com)
            url = f"https://raw.githubusercontent.com/{repo}/HEAD/{file_path}"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                content = resp.text
                # Final size validation (API-reported size can differ from text size)
                if self.target_min <= len(content) <= self.target_max:
                    return content
                # If slightly out of range, try to truncate or pad
                if len(content) > self.target_max:
                    # Truncate at a line boundary
                    content = self._truncate_to_size(content, self.target_max)
                    if content and len(content) >= self.target_min:
                        return content
                elif len(content) < self.target_min:
                    return None  # Too small, skip
            else:
                self.logger.debug(f"Download failed for {repo}/{file_path}: HTTP {resp.status_code}")
        except Exception as e:
            self.logger.debug(f"Download error for {repo}/{file_path}: {e}")
        return None

    @staticmethod
    def _truncate_to_size(content: str, max_bytes: int) -> str:
        """Truncate content at a line boundary to fit within max_bytes."""
        if len(content) <= max_bytes:
            return content
        # Find the last newline before max_bytes
        truncated = content[:max_bytes]
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            return truncated[:last_nl]
        return truncated

    # ------------------------------------------------------------------
    #  Content safety checks
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_text(content: str) -> bool:
        """Basic safety check: reject binary-looking or suspiciously encoded content."""
        if not content:
            return False
        # Check for high ratio of null bytes or non-printable control chars
        suspicious = sum(1 for c in content[:2048] if ord(c) < 9 or (13 < ord(c) < 32))
        if suspicious > len(content[:2048]) * 0.05:
            return False
        return True

    # ------------------------------------------------------------------
    #  Main public API
    # ------------------------------------------------------------------

    def fetch_from_repos(self, repos: list[str] | None = None,
                         max_per_repo: int = 30,
                         max_total: int = 200) -> int:
        """Fetch files from the given (or default) repos into the local cache.

        Skips files already cached and not stale.  Respects API rate limits.

        Args:
            repos:        List of ``owner/repo`` strings.  Defaults to trusted_repos.
            max_per_repo: Max files to download per repo.
            max_total:    Total download cap across all repos.

        Returns:
            Number of newly downloaded files.
        """
        repos = repos or self.trusted_repos
        if not repos:
            self.logger.warning("No repos configured for GitHub file fetching.")
            return 0

        try:
            import requests  # noqa: F811 — verify requests is available
        except ImportError:
            self.logger.error("'requests' package not installed. Cannot fetch GitHub files.")
            return 0

        # Check rate limit before starting
        has_api_budget = self._check_rate_limit()
        if not has_api_budget:
            self.logger.info("API rate limit exhausted — relying on cached files only.")
            return 0

        new_downloads = 0
        repos_shuffled = list(repos)
        random.shuffle(repos_shuffled)

        for repo in repos_shuffled:
            if new_downloads >= max_total:
                break

            if not self._verify_repo_trusted(repo):
                continue

            self.logger.info(f"Fetching file list from {repo}...")
            files = self._list_repo_files(repo, max_files=max_per_repo * 3)
            if not files:
                continue

            # Shuffle and pick up to max_per_repo
            random.shuffle(files)
            fetched_this_repo = 0

            for file_info in files:
                if fetched_this_repo >= max_per_repo or new_downloads >= max_total:
                    break

                fpath = file_info["path"]
                key = self._cache_key(repo, fpath)

                if self._is_cached(key):
                    continue

                content = self._download_file(repo, fpath)
                if content and self._is_safe_text(content):
                    self._write_cache(key, content, repo, fpath)
                    new_downloads += 1
                    fetched_this_repo += 1

                # Small delay to be respectful of GitHub's servers
                time.sleep(0.1)

            self.logger.info(f"  {repo}: fetched {fetched_this_repo} new files")

        self._save_index()
        self.logger.info(f"GitHub fetch complete: {new_downloads} new files downloaded, "
                         f"{len(self._index)} total cached")
        return new_downloads

    def get_cached_count(self) -> int:
        """Number of files currently in the cache (regardless of staleness)."""
        return len(self._index)

    def get_benchmark_items(self, count: int = 20,
                            auto_fetch: bool = True,
                            target_min: int | None = None,
                            target_max: int | None = None) -> list[str]:
        """Return a list of file contents from the cache for use as benchmark items.

        If the cache has fewer than ``count`` items, and ``auto_fetch`` is True,
        this will trigger a fetch from GitHub first.

        Args:
            count:       Number of benchmark items to return.
            auto_fetch:  Whether to auto-download if cache is insufficient.
            target_min:  Override minimum file size (bytes).
            target_max:  Override maximum file size (bytes).

        Returns:
            List of strings (file contents), each within the target size range.
            May return fewer than ``count`` if insufficient files are available.
        """
        effective_min = target_min if target_min is not None else self.target_min
        effective_max = target_max if target_max is not None else self.target_max

        # Auto-fetch if cache is insufficient
        if auto_fetch and self.get_cached_count() < count:
            try:
                self.fetch_from_repos(max_total=max(count * 2, 50))
            except Exception as e:
                self.logger.warning(f"Auto-fetch failed (using cache only): {e}")

        # Collect all valid cached items within size range
        candidates: list[str] = []
        for key, meta in self._index.items():
            cached_size = meta.get("size", 0)
            # Pre-filter by cached metadata size
            if cached_size < effective_min * 0.5 or cached_size > effective_max * 2:
                continue

            content = self._read_cached(key)
            if content is None:
                continue

            # Exact size check on the actual content
            if effective_min <= len(content) <= effective_max:
                if self._is_safe_text(content):
                    candidates.append(content)
            elif len(content) > effective_max:
                # Try to truncate
                truncated = self._truncate_to_size(content, effective_max)
                if truncated and len(truncated) >= effective_min:
                    candidates.append(truncated)

        # Shuffle and return up to count
        random.shuffle(candidates)
        items = candidates[:count]

        self.logger.info(
            f"Returning {len(items)} GitHub benchmark items "
            f"(requested {count}, {len(candidates)} eligible in cache)"
        )
        return items

    def clear_cache(self) -> int:
        """Delete all cached files and the index. Returns number of files removed."""
        removed = 0
        for key in list(self._index.keys()):
            cache_file = os.path.join(self.cache_dir, f"{key}.txt")
            try:
                if os.path.isfile(cache_file):
                    os.remove(cache_file)
                    removed += 1
            except OSError:
                pass
        self._index.clear()
        self._save_index()
        self.logger.info(f"Cache cleared: {removed} files removed.")
        return removed

    def cache_stats(self) -> dict:
        """Return a summary of the cache contents."""
        if not self._index:
            return {"total_files": 0, "total_bytes": 0, "repos": {}}
        total_bytes = sum(m.get("size", 0) for m in self._index.values())
        repos: dict[str, int] = {}
        for meta in self._index.values():
            r = meta.get("repo", "unknown")
            repos[r] = repos.get(r, 0) + 1
        return {
            "total_files": len(self._index),
            "total_bytes": total_bytes,
            "repos": repos,
        }
