from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "ai_prime_capital"
PURE_ROOTS = (SRC_ROOT / "domain", SRC_ROOT / "application")
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "anyio",
        "docx",
        "fastapi",
        "httpx",
        "openai",
        "os",
        "pathlib",
        "pydantic",
        "pydantic_settings",
        "pypdf",
        "qdrant_client",
        "shutil",
        "socket",
        "tempfile",
        "uvicorn",
    }
)


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", maxsplit=1)[0])

    return roots


def pure_python_modules() -> list[Path]:
    return [path for package_root in PURE_ROOTS for path in package_root.rglob("*.py")]


def test_pure_packages_exist_and_contain_python_modules() -> None:
    for package_root in PURE_ROOTS:
        assert package_root.is_dir(), f"missing package boundary: {package_root}"

    modules = pure_python_modules()
    assert modules, "the pure domain/application boundary must not be empty"


def test_domain_and_application_are_vendor_and_io_free() -> None:
    violations: list[str] = []

    for path in pure_python_modules():
        forbidden = imported_roots(path) & FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            violations.append(f"{path}: {sorted(forbidden)}")

    assert not violations, "\n".join(violations)
