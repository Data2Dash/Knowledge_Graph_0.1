import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    log_dir = os.path.join("output", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "debug.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(log_path)
               for h in root.handlers):

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,   # 2MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

        root.addHandler(file_handler)

setup_logging()