import logging
from typing import Any


class Logger:
    def __init__(self, output_dir: str):
        self.logger: logging.Logger = logging.getLogger("")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(output_dir)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, msg: object, *args: object, **kwargs: Any):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: Any):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: Any):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: Any):
        self.logger.error(msg, *args, **kwargs)
