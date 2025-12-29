import logging
from pathlib import Path

def setup_logging(log_path="logs/run.log", level=logging.INFO):
    """
    Configure the logging system.
                
    Parameters
    ----------
    log_path : str, optional
        Path to sabe the log file. Defaults to "logs/run.log".
    level : int, optional
        Logging level. Defaults to logging.INFO.

    Returns
    -------
    None
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Arquivo
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])
