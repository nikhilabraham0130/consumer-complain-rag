"""
Structured and colorized logger for the Consumer Complaint Intelligence RAG system.
Uses the 'rich' library to format log messages with timestamps and log levels.
"""

import logging
import sys
from rich.logging import RichHandler


def get_logger(name: str = "cfpb_rag") -> logging.Logger:
    """
    Returns a configured standard logger with RichHandler formatting.
    
    Args:
        name: Logger name (defaults to 'cfpb_rag')
        
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True
        )
        
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

