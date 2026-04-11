"""HTTP API entrypoints for the forecasting toolkit."""

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name in {"app", "create_app"}:
        from .app import app, create_app

        exports = {"app": app, "create_app": create_app}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
