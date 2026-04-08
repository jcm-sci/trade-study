"""Smoke test — package imports successfully."""


def test_import() -> None:
    """Verify that the package imports and exposes __version__."""
    import trade_study

    assert hasattr(trade_study, "__version__")
