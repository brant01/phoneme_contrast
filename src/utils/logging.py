from pathlib import Path
import logging
from typing import Literal


def create_logger(
    log_dir: Path, console_log_level: Literal["info", "debug"] = "info"
) -> logging.Logger:
    """
    Creates a logger that writes both info-level and debug-level logs to separate files,
    and prints messages to the console at a configurable level.

    Args:
        log_dir (Path): Directory where log files will be saved.
        console_log_level (str): Level of log messages to print to console ("info" or "debug").

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # File handler for INFO level and above
    info_handler = logging.FileHandler(log_dir / "log_info.txt")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    # File handler for DEBUG level (everything)
    debug_handler = logging.FileHandler(log_dir / "log_debug.txt")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if console_log_level == "debug" else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
