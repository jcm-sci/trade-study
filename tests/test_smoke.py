"""Smoke test — package imports successfully."""


def test_import() -> None:
    """Verify that the package imports and exposes __version__."""
    import model_criticism

    assert hasattr(model_criticism, "__version__")
