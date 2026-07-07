import logging
import sys

try:
    from pepe.api import embed  # Exported to top-level; requires torch at runtime
except ImportError:
    pass

# Package metadata - single source of truth
__version__ = "1.3.0"
__package_name__ = "pepe-cli"
__module_name__ = "pepe"
__author__ = "Jahn Zhong"
__author_email__ = "jahn.zhong@medisin.uio.no"
__description__ = "Pipeline for Easy Protein Embedding - Extract embeddings and attention matrices from protein sequences"
__homepage__ = "https://github.com/csi-greifflab/pepe-cli"


# Configure logging for the pepe package
def setup_logging(level=logging.INFO):
    """Set up logging configuration for the PEPE package."""
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get root logger for the package
    logger = logging.getLogger("pepe")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    # Add handler
    logger.addHandler(console_handler)

    return logger


# Set up default logging
setup_logging()
