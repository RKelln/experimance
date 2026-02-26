#!/usr/bin/env python3
"""Check repo-local markdown links in service and library docs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICES_DIR = REPO_ROOT / "services"
LIBS_DIR = REPO_ROOT / "libs"

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")


def _iter_docs() -> list[Path]:
    files: list[Path] = []
    # services/ — README files and docs/ subdirectories
    files.extend(SERVICES_DIR.glob("**/README*.md"))
    files.extend(SERVICES_DIR.glob("**/docs/**/*.md"))
    # libs/ — README files and docs/ subdirectories
    files.extend(LIBS_DIR.glob("**/README*.md"))
    files.extend(LIBS_DIR.glob("**/docs/**/*.md"))
    return [path for path in files if path.is_file()]


def _strip_fragment(link: str) -> str:
    return link.split("#", 1)[0]


def _is_external(link: str) -> bool:
    lower = link.lower()
    return (
        lower.startswith("http://")
        or lower.startswith("https://")
        or lower.startswith("mailto:")
        or "://" in lower
    )


def _resolve_path(link: str, base_dir: Path) -> Path:
    if link.startswith("/"):
        return (REPO_ROOT / link.lstrip("/")).resolve()
    return (base_dir / link).resolve()


def _check_links(file_path: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False

    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        for match in LINK_PATTERN.finditer(line):
            link = match.group(1).strip()
            if not link or link.startswith("#"):
                continue
            if _is_external(link):
                continue

            link_path = _strip_fragment(link)
            if not link_path:
                continue

            resolved = _resolve_path(link_path, file_path.parent)
            if not resolved.exists():
                errors.append(f"{file_path}:{line_number} -> {link}")

    return errors


def main() -> int:
    docs = _iter_docs()
    broken: list[str] = []

    for doc in docs:
        broken.extend(_check_links(doc))

    if broken:
        print("Broken doc links:")
        for item in broken:
            print(f"- {item}")
        return 1

    print(f"Docs links OK ({len(docs)} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
