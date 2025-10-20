import logging
import os
from logging.handlers import RotatingFileHandler

def configure_logging(log_to_console=False):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "app.log")

    # Clear previous log file on boot
    if os.path.exists(log_file):
        os.remove(log_file)

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", errors='ignore'))

    # Configure base logger to only include the file handler
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])

    # Add stream handler only if log_to_console is True
    if log_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(stream_handler)

    LOGGER = logging.getLogger("app.log")
    return LOGGER
