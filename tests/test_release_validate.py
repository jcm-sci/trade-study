"""Tests for release metadata validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.release_validate import (
    citation_version,
    validate_development_metadata,
    validate_release,
    validate_stable_citation,
)


def test_development_versions_match() -> None:
    assert validate_development_metadata() == "0.7.0"


def test_citation_tracks_latest_stable_release() -> None:
    assert citation_version() == "0.7.0"


@pytest.mark.parametrize("version", ["0.7.0.dev0", "0.7", "v0.7.0"])
def test_nonrelease_versions_are_rejected(version: str) -> None:
    with pytest.raises(SystemExit, match="Published version"):
        validate_release(version, f"v{version}")


def test_release_tag_must_match_version() -> None:
    with pytest.raises(SystemExit, match="Release tag"):
        validate_release("0.6.0", "v0.6.1")


def _write_release_fixture(
    tmp_path: Path,
    *,
    doi: str = "10.5281/zenodo.12345",
) -> None:
    tmp_path.joinpath("src/trade_study").mkdir(parents=True)
    tmp_path.joinpath("docs").mkdir()
    tmp_path.joinpath("pyproject.toml").write_text(
        '[project]\nname = "trade-study"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    tmp_path.joinpath("src/trade_study/_version.py").write_text(
        '__version__ = "1.2.3"\n',
        encoding="utf-8",
    )
    tmp_path.joinpath("CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-10-22\n",
        encoding="utf-8",
    )
    tmp_path.joinpath("CITATION.cff").write_text(
        'version: "1.2.3"\n'
        'date-released: "2026-10-22"\n'
        f'doi: "{doi}"\n'
        "identifiers:\n"
        "  - type: doi\n"
        f'    value: "{doi}"\n',
        encoding="utf-8",
    )
    citation = (
        "```bibtex\n"
        "@software{example,\n"
        "  version = {1.2.3},\n"
        f"  doi = {{{doi}}},\n"
        "}\n"
        "```\n"
    )
    tmp_path.joinpath("README.md").write_text(citation, encoding="utf-8")
    tmp_path.joinpath("docs/citation.md").write_text(citation, encoding="utf-8")


def test_release_fixture_checks_exact_doi_and_human_citations(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)

    assert validate_stable_citation(tmp_path) == (
        "1.2.3",
        "10.5281/zenodo.12345",
    )
    assert validate_development_metadata(tmp_path) == "1.2.3"
    validate_release("1.2.3", "v1.2.3", tmp_path)


@pytest.mark.parametrize(
    ("relative", "old", "new", "message"),
    [
        (
            "CITATION.cff",
            'date-released: "2026-10-22"',
            'date-released: "2026-10-21"',
            "date-released",
        ),
        (
            "CITATION.cff",
            'value: "10.5281/zenodo.12345"',
            'value: "10.5281/zenodo.99999"',
            "identifiers",
        ),
        (
            "README.md",
            "10.5281/zenodo.12345",
            "10.5281/zenodo.99999",
            "README.md citation",
        ),
        (
            "docs/citation.md",
            "version = {1.2.3}",
            "version = {1.2.2}",
            "docs/citation.md citation",
        ),
    ],
)
def test_release_fixture_rejects_metadata_drift(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
    message: str,
) -> None:
    _write_release_fixture(tmp_path)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match=message):
        validate_stable_citation(tmp_path)


def test_release_rejects_doi_labelled_for_another_release(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    path = tmp_path / "CITATION.cff"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '    description: "Zenodo archive for release 1.2.2"\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match=r"belongs to release '1\.2\.2'"):
        validate_stable_citation(tmp_path)


def test_release_accepts_doi_labelled_for_same_release(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    path = tmp_path / "CITATION.cff"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '    description: "Zenodo archive for release 1.2.3"\n',
        encoding="utf-8",
    )

    assert validate_stable_citation(tmp_path)[0] == "1.2.3"


def test_pyproject_version_requires_project_table(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    tmp_path.joinpath("pyproject.toml").write_text(
        '[tool.example]\nversion = "1.2.3"\n', encoding="utf-8"
    )

    with pytest.raises(SystemExit, match=r"\[project\] version"):
        validate_development_metadata(tmp_path)
