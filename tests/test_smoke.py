"""Smoke test — package imports successfully."""


def test_import():
    import model_criticism  # noqa: F811

    assert hasattr(model_criticism, "__version__")
