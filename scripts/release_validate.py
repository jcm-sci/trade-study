"""Validate package metadata before a GitHub release can publish to PyPI."""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Final

REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]
RELEASE_VERSION: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?$"
)


def declared_versions(repo_root: pathlib.Path = REPO_ROOT) -> dict[str, str]:
    """Return the versions declared by build and runtime metadata.

    Returns:
        A mapping from metadata file to declared version.

    Raises:
        SystemExit: If the runtime version file has an unsupported format.
    """
    pyproject = (repo_root / "pyproject.toml").read_text("utf-8")
    # A regex instead of tomllib keeps this importable on Python 3.10.
    table = re.search(
        r"^\[project\]\s*$(.*?)(?=^\[|\Z)", pyproject, re.MULTILINE | re.DOTALL
    )
    project = (
        None
        if table is None
        else re.search(r'^version\s*=\s*"([^"]+)"\s*$', table.group(1), re.MULTILINE)
    )
    if project is None:
        msg = "pyproject.toml must declare [project] version as a string."
        raise SystemExit(msg)
    version_source = (repo_root / "src/trade_study/_version.py").read_text("utf-8")
    match = re.fullmatch(r'__version__ = "([^"]+)"\n?', version_source)
    if match is None:
        msg = "src/trade_study/_version.py has an unsupported format."
        raise SystemExit(msg)
    return {
        "pyproject.toml": project.group(1),
        "src/trade_study/_version.py": match.group(1),
    }


def _citation_scalar(key: str, repo_root: pathlib.Path = REPO_ROOT) -> str:
    """Read one top-level scalar from ``CITATION.cff``.

    Returns:
        The unquoted scalar value.

    Raises:
        SystemExit: If the scalar is absent.
    """
    content = (repo_root / "CITATION.cff").read_text("utf-8")
    match = re.search(
        rf"^{re.escape(key)}:\s*[\"']?([^\"'\n]+)[\"']?\s*$",
        content,
        re.MULTILINE,
    )
    if match is None:
        msg = f"CITATION.cff must declare a top-level {key}."
        raise SystemExit(msg)
    return match.group(1).strip()


def citation_version(repo_root: pathlib.Path = REPO_ROOT) -> str:
    """Read the stable release version from CITATION.cff.

    Returns:
        The cited stable version.
    """
    return _citation_scalar("version", repo_root)


def _changelog_date(version: str, repo_root: pathlib.Path = REPO_ROOT) -> str:
    """Return the release date declared for ``version`` in the changelog.

    Returns:
        The ISO release date.

    Raises:
        SystemExit: If the changelog has no matching dated release.
    """
    content = (repo_root / "CHANGELOG.md").read_text("utf-8")
    match = re.search(
        rf"^## \[{re.escape(version)}\] [-—] (\d{{4}}-\d{{2}}-\d{{2}})$",
        content,
        re.MULTILINE,
    )
    if match is None:
        msg = f"CHANGELOG.md has no dated [{version}] release section."
        raise SystemExit(msg)
    return match.group(1)


def _bibtex_software_fields(path: pathlib.Path) -> dict[str, str]:
    """Read version and DOI from the first software BibTeX block.

    Returns:
        The human-readable citation's version and DOI.

    Raises:
        SystemExit: If the block or either required field is absent.
    """
    content = path.read_text("utf-8")
    block = re.search(r"```bibtex\s+@software\{.*?\n\}\s+```", content, re.DOTALL)
    if block is None:
        msg = f"{path.relative_to(path.parents[1])} has no software BibTeX block."
        raise SystemExit(msg)
    fields: dict[str, str] = {}
    for key in ("version", "doi"):
        match = re.search(
            rf"^\s*{key}\s*=\s*\{{([^}}]+)\}}", block.group(), re.MULTILINE
        )
        if match is None:
            msg = f"{path.name} software citation must declare {key}."
            raise SystemExit(msg)
        fields[key] = match.group(1).strip()
    return fields


def validate_stable_citation(repo_root: pathlib.Path = REPO_ROOT) -> tuple[str, str]:
    """Require the stable citation to agree across metadata and prose.

    Returns:
        Stable version and exact release DOI.

    Raises:
        SystemExit: If dates, DOI, or human-readable citations drift.
    """
    version = citation_version(repo_root)
    release_date = _citation_scalar("date-released", repo_root)
    doi = _citation_scalar("doi", repo_root)
    expected_date = _changelog_date(version, repo_root)
    if release_date != expected_date:
        msg = (
            f"CITATION.cff date-released {release_date!r} does not match "
            f"CHANGELOG.md {expected_date!r}."
        )
        raise SystemExit(msg)
    citation = (repo_root / "CITATION.cff").read_text("utf-8")
    if f'value: "{doi}"' not in citation:
        msg = "CITATION.cff identifiers must include its exact top-level release DOI."
        raise SystemExit(msg)
    # Zenodo mints a version DOI only after publication, so a tagged commit may
    # cite the concept DOI, but never a DOI labelled for a different release.
    labelled = re.search(
        rf'value: "{re.escape(doi)}"\s*\n\s*description: "[^"\n]*release '
        r'([0-9][^"\s]*)"',
        citation,
    )
    if labelled is not None and labelled.group(1) != version:
        msg = (
            f"CITATION.cff top-level DOI {doi!r} belongs to release "
            f"{labelled.group(1)!r}, not {version!r}; cite the concept DOI until "
            "Zenodo mints the new release DOI."
        )
        raise SystemExit(msg)
    for relative in ("README.md", "docs/citation.md"):
        fields = _bibtex_software_fields(repo_root / relative)
        if fields != {"version": version, "doi": doi}:
            msg = (
                f"{relative} citation {fields!r} does not match "
                f"CITATION.cff version={version!r}, doi={doi!r}."
            )
            raise SystemExit(msg)
    return version, doi


def validate_development_metadata(repo_root: pathlib.Path = REPO_ROOT) -> str:
    """Require build and import-time versions to agree.

    Returns:
        The shared development version.

    Raises:
        SystemExit: If build and runtime versions differ.
    """
    versions = declared_versions(repo_root)
    if len(set(versions.values())) != 1:
        rendered = ", ".join(f"{path}={value}" for path, value in versions.items())
        msg = f"Package versions do not match: {rendered}"
        raise SystemExit(msg)
    validate_stable_citation(repo_root)
    return next(iter(versions.values()))


def validate_release(
    version: str,
    release_tag: str,
    repo_root: pathlib.Path = REPO_ROOT,
) -> None:
    """Require canonical release metadata and an exact matching tag.

    Raises:
        SystemExit: If the version, tag, or stable citation metadata disagree.
    """
    if RELEASE_VERSION.fullmatch(version) is None:
        msg = (
            "Published version must be canonical X.Y.Z or X.Y.ZaN/bN/rcN; "
            f"got {version!r}."
        )
        raise SystemExit(msg)
    expected_tag = f"v{version}"
    if release_tag != expected_tag:
        msg = f"Release tag must be {expected_tag!r}, got {release_tag!r}."
        raise SystemExit(msg)
    cff_version, _ = validate_stable_citation(repo_root)
    if cff_version != version:
        msg = f"CITATION.cff version {cff_version!r} does not match {version!r}."
        raise SystemExit(msg)


def main() -> None:
    """Validate development metadata and, when requested, a release tag."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-tag")
    args = parser.parse_args()
    version = validate_development_metadata()
    if args.release_tag is not None:
        validate_release(version, args.release_tag)
    print(f"Validated package metadata: {version}")


if __name__ == "__main__":
    main()
