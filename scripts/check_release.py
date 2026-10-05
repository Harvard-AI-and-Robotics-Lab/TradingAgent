#!/usr/bin/env python3
"""Offline safety and structure checks for the public release tree."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", ".venv", "venv", "build", "dist", "__pycache__"}
TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REQUIRED = {
    "README.md",
    "assets/trusttrade_framework.png",
    "assets/risk_return_tradeoff_2024q1.png",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "SECURITY.md",
    "pyproject.toml",
    "main.py",
    "docs/DATA.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/REPRODUCIBILITY.md",
    "docs/UPSTREAM.md",
    "configs/models.template.json",
    "configs/experiments/2024_q1.example.json",
    "configs/experiments/2026_q1.example.json",
}
SECRET_PATTERNS = {
    "OpenAI/Anthropic-style key": re.compile("s" + r"k-[A-Za-z0-9_-]{20,}"),
    "Google API key": re.compile("AI" + r"za[0-9A-Za-z_-]{30,}"),
    "xAI key": re.compile("x" + r"ai-[A-Za-z0-9_-]{30,}"),
    "AWS access key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
    "GitHub token": re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    "private key block": re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY"),
}
PRIVATE_PATH = re.compile(r"/(?:home|Users|PHShome)/[A-Za-z0-9._-]+/")
PERSON_FIELDS = (
    "education" + "Level",
    "trading" + "Experience",
    "isBusiness" + "Student",
)


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    errors: list[str] = []

    missing = sorted(name for name in REQUIRED if not (ROOT / name).is_file())
    errors.extend(f"missing required file: {name}" for name in missing)

    notebooks = [path.relative_to(ROOT) for path in ROOT.rglob("*.ipynb")]
    errors.extend(f"notebook output is not allowed: {path}" for path in notebooks)

    python_count = 0
    json_count = 0
    for path in iter_files():
        relative = path.relative_to(ROOT)
        if path.suffix == ".py":
            python_count += 1
            try:
                compile(path.read_text(encoding="utf-8"), str(relative), "exec")
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python compile failure in {relative}: {exc}")

        if path.suffix == ".json":
            json_count += 1
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                errors.append(f"invalid JSON in {relative}: {exc}")

        if path.suffix not in TEXT_SUFFIXES and path.name not in {"NOTICE", ".env.example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {relative}")
        if PRIVATE_PATH.search(text):
            errors.append(f"private machine path in {relative}")
        if relative.parts[:2] == ("baseline_analysis", "human"):
            errors.append(f"person-level human artifact in {relative}")
        if path.suffix == ".json" and any(field in text for field in PERSON_FIELDS):
            errors.append(f"person-level demographic field in {relative}")

    if errors:
        print("Release validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Release validation passed: "
        f"{python_count} Python files and {json_count} JSON files checked; "
        "no credential patterns or person-level records detected."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
