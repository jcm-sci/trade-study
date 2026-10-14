"""Immutable result generations selected by one atomic local-filesystem pointer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple
from uuid import uuid4


class _Generation(NamedTuple):
    directory: Path
    metadata: bytes
    identifier: str


def _resolve_generation(root: Path) -> _Generation | None:
    """Capture a single published pointer and its checked metadata bytes.

    Returns:
        One immutable generation, or None for an unpublished/direct destination.

    Raises:
        ValueError: If the pointer or selected generation metadata is inconsistent.
    """
    pointer_path = root / "current.json"
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_bytes())
    if (
        not isinstance(pointer, dict)
        or set(pointer) != {"schema_version", "generation", "manifest_sha256"}
        or type(pointer["schema_version"]) is not int
        or pointer["schema_version"] != 1
    ):
        msg = "Unsupported or incomplete result-publication pointer"
        raise ValueError(msg)
    identifier = pointer["generation"]
    if (
        not isinstance(identifier, str)
        or len(identifier) != 32
        or not set(identifier) <= set("0123456789abcdef")
    ):
        msg = "Publication generation must be a 32-character lowercase hex identifier"
        raise ValueError(msg)
    directory = root / "generations" / identifier
    metadata = (directory / "meta.json").read_bytes()
    if pointer["manifest_sha256"] != hashlib.sha256(metadata).hexdigest():
        msg = "Published generation metadata does not match its pointer digest"
        raise ValueError(msg)
    manifest = json.loads(metadata)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        msg = "Published generations require version-1 result metadata"
        raise ValueError(msg)
    return _Generation(directory, metadata, identifier)


def _new_generation(root: Path) -> Path:
    """Reserve a unique generation namespace without changing a current pointer.

    Returns:
        A new not-yet-created data directory for the exclusive schema writer.
    """
    _resolve_generation(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "generations").mkdir(exist_ok=True)
    return root / "generations" / uuid4().hex


def _commit_generation(root: Path, generation: Path) -> None:
    """Publish a fully written generation in one same-directory atomic replace."""
    pointer = {
        "schema_version": 1,
        "generation": generation.name,
        "manifest_sha256": hashlib.sha256(
            (generation / "meta.json").read_bytes()
        ).hexdigest(),
    }
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, prefix=".current-", delete=False
    ) as temporary:
        temporary.write(json.dumps(pointer, allow_nan=False))
        temporary_path = Path(temporary.name)
    temporary_path.replace(root / "current.json")
