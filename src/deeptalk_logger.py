import logging
from termcolor import colored


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }

    def format(self, record):
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname, "white"))


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s %(name)s[%(levelname)s] %(message)s"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[handler],
)

class DeepTalkLogger:
    def __init__(self):
        self.logger = logging.getLogger()

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def debug(self, msg, *args, **kwargs):
        try:
            self.logger.debug(msg, *args, **kwargs)
        finally:
            pass

    def info(self, msg, *args, **kwargs):
        try:
            self.logger.info(msg, *args, **kwargs)
        finally:
            pass

    def warning(self, msg, *args, **kwargs):
        try:
            self.logger.warning(msg, *args, **kwargs)
        finally:
            pass

    def error(self, msg, *args, **kwargs):
        try:
            self.logger.error(msg, *args, **kwargs)
        finally:
            pass

    def critical(self, msg, *args, **kwargs):
        try:
            self.logger.critical(msg, *args, **kwargs)
        finally:
            pass
