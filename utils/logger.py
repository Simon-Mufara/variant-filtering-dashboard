"""Centralised logging for the variant dashboard."""
import logging
import sys


def get_logger(name: str = "variant_dashboard") -> logging.Logger:
    """Return a configured logger. Calling this multiple times with the same
    name always returns the same logger instance (Python guarantee).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


log = get_logger()
