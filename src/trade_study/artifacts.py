"""Validated result-bundle schemas and primitive JSON input preservation."""

from __future__ import annotations

import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

from ._version import __version__
from .protocols import Direction, Observable, ResultsTable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from numpy.typing import NDArray


class ResultsBundle(NamedTuple):
    """Scored data plus separately preserved scientific interpretation.

    Attributes:
        results: Reconstructed parallel score/config/annotation table.
        observables: Ordered directions and weights; empty for legacy exports.
        provenance: Caller identities and recorded runtime versions. This does
            not reconstruct executable simulators/scorers or certify replay.
        schema_version: One for primitive JSON bundles, zero for legacy export.
    """

    results: ResultsTable
    observables: tuple[Observable, ...]
    provenance: Mapping[str, object]
    schema_version: int


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        msg = f"{name} must be a mapping with string keys"
        raise ValueError(msg)
    return cast("dict[str, object]", value)


def _json_value(value: object) -> object:
    """Detach primitive JSON values without stringifying unsupported objects.

    Returns:
        Fresh nested containers preserving supported primitive values.

    Raises:
        ValueError: If values are non-finite or not primitive JSON types.
    """
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _json_value(item)
            for key, item in _mapping(value, "JSON value").items()
        }
    msg = f"Unsupported replay-storage value of type {type(value).__qualname__}"
    raise ValueError(msg)


def _names(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        msg = f"{name} must be a list of nonempty names"
        raise ValueError(msg)
    if len(set(value)) != len(value):
        msg = f"{name} must contain unique names"
        raise ValueError(msg)
    return cast("list[str]", value)


def _rows(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        msg = f"{name} must be a list of row mappings"
        raise TypeError(msg)
    for row in value:
        _mapping(row, name)
    return cast("list[dict[str, Any]]", value)


def _validate_table(results: ResultsTable) -> None:
    names = _names(results.observable_names, "observable_names")
    annotations = _names(results.annotation_names, "annotation_names")
    rows = _rows(results.configs, "configs")
    metadata = _rows(results.metadata, "metadata")
    scores = results.scores
    if (
        not isinstance(scores, np.ndarray)
        or scores.dtype.kind not in "fiu"
        or scores.ndim != 2
        or scores.shape != (len(rows), len(names))
    ):
        msg = (
            "scores must be a real numeric matrix matching config rows "
            "and observable names"
        )
        raise ValueError(msg)
    if metadata and len(metadata) != len(rows):
        msg = "metadata must be empty or contain one mapping per config row"
        raise ValueError(msg)
    if results.annotations is None:
        if annotations:
            msg = "annotation_names require an annotation matrix"
            raise ValueError(msg)
    elif (
        not isinstance(results.annotations, np.ndarray)
        or results.annotations.dtype.kind not in "fiu"
        or results.annotations.ndim != 2
        or results.annotations.shape != (len(rows), len(annotations))
    ):
        msg = (
            "annotations must be a real numeric matrix matching rows "
            "and annotation names"
        )
        raise ValueError(msg)


def _observable_metadata(
    observables: Sequence[Observable], names: list[str]
) -> list[dict[str, object]]:
    if [observable.name for observable in observables] != names:
        msg = "observable definitions must match score names in order"
        raise ValueError(msg)
    result: list[dict[str, object]] = []
    for observable in observables:
        if (
            not isinstance(observable.direction, Direction)
            or isinstance(observable.weight, bool)
            or not isinstance(observable.weight, Real)
            or not math.isfinite(observable.weight)
            or observable.weight < 0
        ):
            msg = "observable directions and finite nonnegative weights are required"
            raise ValueError(msg)
        result.append({
            "name": observable.name,
            "direction": observable.direction.value,
            "weight": float(observable.weight),
        })
    return result


def _provenance(value: object) -> dict[str, object]:
    provenance = _mapping(value, "provenance")
    if set(provenance) != {"runtime", "caller"}:
        msg = "provenance must contain runtime versions and caller identities"
        raise ValueError(msg)
    runtime = _mapping(provenance["runtime"], "runtime provenance")
    if set(runtime) != {"trade-study", "numpy"} or any(
        not isinstance(version, str) or not version for version in runtime.values()
    ):
        msg = "runtime provenance must record trade-study and NumPy version strings"
        raise ValueError(msg)
    _mapping(provenance["caller"], "caller provenance")
    return provenance


def _array_description(
    path: Path, array: NDArray[np.floating[Any]]
) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _read_array(
    path: Path, key: str, descriptor: object | None = None
) -> NDArray[np.floating[Any]]:
    """Read one array from the same bytes checked against its declared digest.

    Returns:
        Materialized array without loading pickled objects.

    Raises:
        ValueError: If files, schema descriptions or array contents disagree.
    """
    payload = path.read_bytes()
    description = (
        None if descriptor is None else _mapping(descriptor, "array description")
    )
    if description is not None and (
        set(description) != {"filename", "sha256", "shape", "dtype"}
        or description["filename"] != path.name
        or description["sha256"] != hashlib.sha256(payload).hexdigest()
    ):
        msg = f"Array file {path.name!r} does not match its manifest"
        raise ValueError(msg)
    if description is not None:
        shape = description["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            msg = "array shape metadata must contain two nonnegative integers"
            raise ValueError(msg)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if archive.files != [key]:
            msg = f"Array file {path.name!r} must contain exactly {key!r}"
            raise ValueError(msg)
        array = archive[key]
    if description is not None and (
        description["shape"] != list(array.shape)
        or description["dtype"] != array.dtype.str
    ):
        msg = f"Array file {path.name!r} has inconsistent shape or dtype metadata"
        raise ValueError(msg)
    return cast("NDArray[np.floating[Any]]", array)


def _table(
    meta: Mapping[str, object],
    scores: NDArray[np.floating[Any]],
    annotations: NDArray[np.floating[Any]] | None,
) -> ResultsTable:
    result = ResultsTable(
        configs=_rows(meta["configs"], "configs"),
        scores=scores,
        observable_names=_names(meta["observable_names"], "observable_names"),
        annotations=annotations,
        annotation_names=_names(meta.get("annotation_names", []), "annotation_names"),
        metadata=_rows(meta.get("metadata", []), "metadata"),
    )
    _validate_table(result)
    return result


def _read_v1(path: Path, meta: dict[str, object]) -> ResultsBundle:
    if (
        set(meta)
        != {
            "schema_version",
            "encoding",
            "observable_names",
            "annotation_names",
            "configs",
            "metadata",
            "observables",
            "provenance",
            "arrays",
        }
        or type(meta["schema_version"]) is not int
        or meta["schema_version"] != 1
        or meta["encoding"] != "primitive-json-v1"
    ):
        msg = "Unsupported or incomplete result-bundle schema"
        raise ValueError(msg)
    _json_value(meta)
    arrays = _mapping(meta["arrays"], "arrays")
    if set(arrays) != {"scores", "annotations"} or arrays["scores"] is None:
        msg = "Result-bundle arrays must declare scores and optional annotations"
        raise ValueError(msg)
    scores = _read_array(path / "scores.npz", "scores", arrays["scores"])
    annotations = None
    if arrays["annotations"] is not None:
        annotations = _read_array(
            path / "annotations.npz", "annotations", arrays["annotations"]
        )
    elif (path / "annotations.npz").exists():
        msg = "Undeclared annotation file in result bundle"
        raise ValueError(msg)
    results = _table(meta, scores, annotations)
    raw = meta["observables"]
    if not isinstance(raw, list):
        msg = "observable definitions must be a list"
        raise TypeError(msg)
    observables = []
    for value in raw:
        definition = _mapping(value, "observable definition")
        if set(definition) != {"name", "direction", "weight"}:
            msg = "observable definitions need name, direction and weight"
            raise ValueError(msg)
        observables.append(
            Observable(
                cast("str", definition["name"]),
                Direction(definition["direction"]),
                cast("float", definition["weight"]),
            )
        )
    _observable_metadata(observables, results.observable_names)
    return ResultsBundle(
        results, tuple(observables), _provenance(meta["provenance"]), 1
    )


def _load_bundle(directory: Path) -> ResultsBundle:
    meta = _mapping(
        json.loads((directory / "meta.json").read_text(encoding="utf-8")),
        "result metadata",
    )
    if "schema_version" in meta:
        return _read_v1(directory, meta)
    if set(meta) - {"configs", "observable_names", "annotation_names", "metadata"}:
        msg = "Unrecognized legacy metadata; a schema_version may be missing"
        raise ValueError(msg)
    scores = _read_array(directory / "scores.npz", "scores")
    ann_path = directory / "annotations.npz"
    annotations = _read_array(ann_path, "annotations") if ann_path.exists() else None
    return ResultsBundle(_table(meta, scores, annotations), (), {}, 0)


def load_result_bundle(path: str | Path) -> ResultsBundle:
    """Load a validated version-1 bundle or a consistent legacy export.

    Args:
        path: Directory containing a committed meta.json and array files.

    Returns:
        Scored table and available scientific metadata. Legacy schema version
        zero has no preserved observable definitions or replayability claim.

    Raises:
        ValueError: If schema, array digests, parallel rows or metadata disagree.
    """
    try:
        return _load_bundle(Path(path))
    except (ValueError, KeyError, TypeError) as error:
        msg = f"Invalid result bundle: {error}"
        raise ValueError(msg) from error


def save_result_bundle(
    results: ResultsTable,
    path: str | Path,
    *,
    observables: Sequence[Observable],
    provenance: dict[str, object] | None = None,
) -> None:
    """Write a version-1 primitive JSON bundle into a new destination only.

    Validate all values before creating the destination. Reserve a new directory,
    write array snapshots, and atomically publish metadata last. An interrupted
    write has no committed metadata and cannot read as a complete bundle. This
    initial schema writer does not overwrite or update existing directories.

    Args:
        results: Parallel scored data; arrays may contain non-finite score values.
        path: New destination directory, on a filesystem supporting atomic replace.
        observables: Ordered scientific directions/weights matching score columns.
        provenance: Primitive JSON identities for model/scorer/transforms, external
            inputs, replication and random streams. Runtime versions are added.

    Raises:
        ValueError: If data or interpretation is inconsistent or values unsupported.
        FileExistsError: If the destination already exists, including incomplete data.
    """
    try:
        _validate_table(results)
        meta = {
            "schema_version": 1,
            "encoding": "primitive-json-v1",
            "observable_names": list(results.observable_names),
            "annotation_names": list(results.annotation_names),
            "configs": _json_value(results.configs),
            "metadata": _json_value(results.metadata),
            "observables": _observable_metadata(observables, results.observable_names),
            "provenance": {
                "runtime": {"trade-study": __version__, "numpy": np.__version__},
                "caller": _json_value(
                    _mapping(
                        {} if provenance is None else provenance, "caller provenance"
                    )
                ),
            },
        }
        scores = results.scores.copy()
        annotations = (
            None if results.annotations is None else results.annotations.copy()
        )
    except (ValueError, TypeError, RecursionError) as error:
        msg = f"Cannot save replayable result inputs: {error}"
        raise ValueError(msg) from error
    directory = Path(path)
    if directory.exists():
        msg = "Result-bundle destination must be new; existing data is not overwritten"
        raise FileExistsError(msg)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "scores.npz", scores=scores)
    annotation_description = None
    if annotations is not None:
        np.savez_compressed(directory / "annotations.npz", annotations=annotations)
        annotation_description = _array_description(
            directory / "annotations.npz", annotations
        )
    meta["arrays"] = {
        "scores": _array_description(directory / "scores.npz", scores),
        "annotations": annotation_description,
    }
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=".meta-", delete=False
    ) as temporary:
        temporary.write(json.dumps(meta, indent=2, allow_nan=False))
        temporary_path = Path(temporary.name)
    temporary_path.replace(directory / "meta.json")
