# encoding: utf-8
"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def get_logger(name: str = "regression_analysis") -> logger:
    """Get a configured logger instance.

    Removes default loguru handler and adds:
    - Console handler at INFO level
    - File handler at DEBUG level to logs/ directory
    """
    logger.remove()

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=True,
    )

    logger.add(
        log_dir / "regression_analysis.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
    )

    return logger
