"""Smoke test — package imports successfully."""


def test_import():
    import PACKAGE_NAME  # noqa: F811

    assert hasattr(PACKAGE_NAME, "__version__")
