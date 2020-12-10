import logging
import os
from pathlib import Path


def init_logger(
    name: str = "dsvae", level: int = logging.INFO
) -> logging.Logger:

    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger.setLevel(level)
        handler = logging.StreamHandler()
        format = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        handler.setFormatter(format)
        handler.setLevel(level)
        logger.addHandler(handler)
    return logger


def load_path() -> Path:
    try:
        path = Path(os.environ["PATH_TO_DATA"])
        return path
    except KeyError:
        raise EnvironmentError(
            "Please set to PATH_TO_DATA as an environtment variable"
            "pointing to your data directory."
        )
