"""Logging setup shared across all modules."""

import logging
import sys


class _BoundLogger:
    """Simple wrapper so callers can pass keyword args like structlog."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _fmt(self, msg: str, kwargs: dict) -> str:
        if kwargs:
            kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{msg} | {kv}"
        return msg

    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(self._fmt(msg, kwargs))

    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(self._fmt(msg, kwargs))

    def error(self, msg: str, **kwargs) -> None:
        self._logger.error(self._fmt(msg, kwargs))

    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(self._fmt(msg, kwargs))


def get_logger(name: str) -> _BoundLogger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return _BoundLogger(logger)
