"""
Logging configuration module.
"""
import sys
from loguru import logger
from pathlib import Path
from config.settings import settings

def setup_logger():
    """Configure application logging"""
    # Remove default console handlers
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level="INFO" if not settings.DEBUG else "DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler
    log_file = settings.LOGS_DIR / "gis_rag.log"
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger
    
# Global logger instance
log = setup_logger()




