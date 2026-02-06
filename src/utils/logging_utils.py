"""
Logging Utilities

Provides consistent logging setup across the project.

Usage:
    from src.utils.logging_utils import setup_logging, get_logger
    
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file path for logging output
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Helper for logging progress in long-running operations."""
    
    def __init__(
        self, 
        name: str,
        total: int,
        log_interval: int = 100
    ):
        """
        Args:
            name: Logger name
            total: Total number of items
            log_interval: How often to log progress
        """
        self.logger = get_logger(name)
        self.total = total
        self.log_interval = log_interval
        self.current = 0
        self.start_time = None
    
    def start(self):
        """Start progress tracking."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting processing of {self.total} items")
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"Progress: {self.current}/{self.total} "
                f"({100*self.current/self.total:.1f}%) - "
                f"{rate:.1f} items/sec - "
                f"ETA: {remaining:.0f}s"
            )
    
    def finish(self):
        """Finish progress tracking."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"Completed {self.total} items in {elapsed:.1f}s "
            f"({self.total/elapsed:.1f} items/sec)"
        )


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler compatible with tqdm progress bars.
    
    Prevents log messages from breaking tqdm output.
    """
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_colab_logging():
    """Set up logging optimized for Google Colab."""
    setup_logging(
        level=logging.INFO,
        format_string='%(levelname)s: %(message)s'
    )
    
    # Also configure for notebook display
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

