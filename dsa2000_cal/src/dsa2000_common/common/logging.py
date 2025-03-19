import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # This ensures that loggers from external libraries (like Ray) are not disabled
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d->%(funcName)s) %(name)s: %(message)s",
        },
        "verbose": {
            "format": "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d->%(funcName)s) %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "verbose",
            "filename": "application.log",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        "ray": {  # For Ray's logger
            "level": "INFO",
            "handlers": ["console"],
            "propagate": True,
        },
        "transitions.extensions.asyncio": {
            "level": "WARNING",  # Only show WARNING or higher for transitions
            "handlers": ["console"],
            "propagate": False,
        },
        # You can add more module-specific configurations here
        "dsa2000_cal": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        }
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
dsa_logger = logging.getLogger("dsa2000_cal")


def test_logging():
    dsa_logger.info("This is an info message")
    dsa_logger.debug("This is a debug message")
    dsa_logger.warning("This is a warning message")
