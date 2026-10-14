"""Publication failures, captured-reader generations and concurrent writers."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import numpy as np
import pytest

from trade_study import (
    Direction,
    Observable,
    ResultsTable,
    load_result_bundle,
    load_results,
    publish_result_bundle,
    save_result_bundle,
    save_results,
)


def _results(value: float) -> ResultsTable:
    return ResultsTable(
        configs=[{"value": value}],
        scores=np.array([[value]]),
        observable_names=["metric"],
        metadata=[{"run": value}],
    )


def _publish(root: Path, value: float) -> Path:
    return publish_result_bundle(
        _results(value),
        root,
        observables=[Observable("metric", Direction.MINIMIZE)],
        provenance={"run": value},
    )


def test_published_generations_are_complete_retained_and_separately_addressable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    first = _publish(root, 1.0)
    old = load_result_bundle(root)
    second = _publish(root, 2.0)
    current = load_result_bundle(root)
    assert first != second
    assert old.generation == first.name
    assert current.generation == second.name
    assert old.results.configs == [{"value": 1.0}]
    assert current.results.configs == [{"value": 2.0}]
    assert current.results.metadata == [{"run": 2.0}]
    assert current.provenance["caller"] == {"run": 2.0}
    np.testing.assert_array_equal(old.results.scores, [[1.0]])
    np.testing.assert_array_equal(load_results(root).scores, [[2.0]])
    np.testing.assert_array_equal(load_result_bundle(first).results.scores, [[1.0]])
    assert load_result_bundle(first).generation is None
    assert set(json.loads((root / "current.json").read_text())) == {
        "schema_version",
        "generation",
        "manifest_sha256",
    }


def test_failed_pointer_replacement_retains_previous_complete_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results"
    first = _publish(root, 1.0)
    previous = (root / "current.json").read_bytes()
    original = Path.replace

    def fail_current(path: Path, target: Path) -> Path:
        if target.name == "current.json":
            msg = "synthetic pointer publication failure"
            raise OSError(msg)
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_current)
    with pytest.raises(OSError, match="publication failure"):
        _publish(root, 2.0)
    assert (root / "current.json").read_bytes() == previous
    assert load_result_bundle(root).generation == first.name
    np.testing.assert_array_equal(load_results(root).scores, [[1.0]])
    assert len(list((root / "generations").iterdir())) == 2


def test_failed_array_write_keeps_active_generation_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results"
    first = _publish(root, 1.0)
    pointer = (root / "current.json").read_bytes()

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        msg = "synthetic array write failure"
        raise OSError(msg)

    monkeypatch.setattr(np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="array write failure"):
        _publish(root, 2.0)
    assert (root / "current.json").read_bytes() == pointer
    assert load_result_bundle(root).generation == first.name
    np.testing.assert_array_equal(load_results(root).scores, [[1.0]])


def test_reader_keeps_one_captured_pointer_during_an_overlapping_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results"
    first = _publish(root, 1.0)
    original = Path.read_bytes
    committed = False

    def read_then_publish(path: Path) -> bytes:
        nonlocal committed
        captured = original(path)
        if path == root / "current.json" and not committed:
            committed = True
            _publish(root, 2.0)
        return captured

    monkeypatch.setattr(Path, "read_bytes", read_then_publish)
    captured = load_result_bundle(root)
    assert captured.generation == first.name
    assert captured.results.configs == [{"value": 1.0}]
    assert captured.results.metadata == [{"run": 1.0}]
    np.testing.assert_array_equal(captured.results.scores, [[1.0]])
    latest = load_result_bundle(root)
    assert latest.results.configs == [{"value": 2.0}]
    np.testing.assert_array_equal(latest.results.scores, [[2.0]])


def test_concurrent_writers_publish_only_complete_distinct_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results"
    _publish(root, 0.0)
    barrier = Barrier(2)
    original = Path.replace

    def replace_together(path: Path, target: Path) -> Path:
        if target == root / "current.json":
            barrier.wait(timeout=10)
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace_together)
    with ThreadPoolExecutor(max_workers=2) as workers:
        generations = list(workers.map(lambda value: _publish(root, value), [1.0, 2.0]))
    assert generations[0] != generations[1]
    for generation, value in zip(generations, [1.0, 2.0], strict=True):
        np.testing.assert_array_equal(
            load_result_bundle(generation).results.scores, [[value]]
        )
    current = load_result_bundle(root)
    assert current.generation in {generation.name for generation in generations}
    value = current.results.configs[0]["value"]
    assert current.results.metadata == [{"run": value}]
    assert current.provenance["caller"] == {"run": value}
    np.testing.assert_array_equal(current.results.scores, [[value]])


@pytest.mark.parametrize("versioned", [False, True])
def test_publication_retains_direct_legacy_or_versioned_files(
    tmp_path: Path, *, versioned: bool
) -> None:
    root = tmp_path / "results"
    if versioned:
        save_result_bundle(
            _results(1.0), root, observables=[Observable("metric", Direction.MINIMIZE)]
        )
    else:
        save_results(_results(1.0), root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    _publish(root, 2.0)
    assert {name: (root / name).read_bytes() for name in before} == before
    np.testing.assert_array_equal(load_results(root).scores, [[2.0]])
    with pytest.raises(ValueError, match="cannot overwrite a published"):
        save_results(_results(3.0), root)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"generation": "../other"},
        {"manifest_sha256": "incorrect"},
        {"generation": "0" * 32},
    ],
)
def test_inconsistent_pointer_never_falls_back_to_old_legacy_files(
    tmp_path: Path, change: dict[str, object]
) -> None:
    root = tmp_path / "results"
    save_results(_results(0.0), root)
    _publish(root, 1.0)
    pointer_path = root / "current.json"
    pointer = json.loads(pointer_path.read_bytes())
    pointer.update(change)
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_result_bundle(root)


def test_modified_generation_metadata_fails_pointer_integrity(tmp_path: Path) -> None:
    root = tmp_path / "results"
    generation = _publish(root, 1.0)
    metadata = generation / "meta.json"
    contents = json.loads(metadata.read_bytes())
    contents["configs"] = [{"value": 2.0}]
    metadata.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(ValueError, match="pointer digest"):
        load_result_bundle(root)


def test_invalid_existing_current_file_is_preserved(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    pointer_path = root / "current.json"
    pointer_path.write_text("unrelated or corrupt data", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot publish"):
        _publish(root, 1.0)
    assert pointer_path.read_text(encoding="utf-8") == "unrelated or corrupt data"
    assert not (root / "generations").exists()
