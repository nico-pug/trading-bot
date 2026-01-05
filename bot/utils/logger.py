"""
Professional Logging System for Trading Bot
Sostituisce tutti i print() con un sistema di logging strutturato.

Usage:
    from bot.utils.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info("Trade eseguito")
    logger.warning("Drawdown elevato")
    logger.error("Connessione API fallita", exc_info=True)
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional

# Directory per i log
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'logs')

# Colori per console (Windows compatible)
class ColoredFormatter(logging.Formatter):
    """Formatter con colori per console"""
    
    # Codici ANSI per colori
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        # Abilita colori su Windows
        if use_colors and os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                self.use_colors = False
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


class TradeLogFormatter(logging.Formatter):
    """Formatter specializzato per log di trading"""
    
    def __init__(self):
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        # Aggiungi contesto trading se presente
        if hasattr(record, 'symbol'):
            record.msg = f"[{record.symbol}] {record.msg}"
        if hasattr(record, 'trade_id'):
            record.msg = f"(#{record.trade_id}) {record.msg}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console: bool = True,
    use_colors: bool = True
) -> None:
    """
    Configura il sistema di logging globale.
    
    Args:
        level: Livello minimo di log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Nome file log (default: bot_YYYYMMDD.log)
        max_bytes: Dimensione massima file prima di rotazione
        backup_count: Numero di file di backup da mantenere
        console: Se mostrare log in console
        use_colors: Se usare colori in console
    """
    # Crea directory logs se non esiste
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Rimuovi handler esistenti
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Handler per console
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
        console_handler.setFormatter(ColoredFormatter(console_fmt, "%H:%M:%S", use_colors))
        root_logger.addHandler(console_handler)
    
    # Handler per file con rotazione
    if log_file is None:
        log_file = f"bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_path = os.path.join(LOG_DIR, log_file)
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(TradeLogFormatter())
    root_logger.addHandler(file_handler)
    
    # Log iniziale
    root_logger.info(f"Logging inizializzato - File: {file_path}")


def get_logger(name: str, symbol: Optional[str] = None) -> logging.Logger:
    """
    Ottiene un logger con nome specifico.
    
    Args:
        name: Nome del modulo (tipicamente __name__)
        symbol: Simbolo crypto opzionale per contestualizzare i log
        
    Returns:
        Logger configurato
    """
    logger = logging.getLogger(name)
    
    # Aggiungi adattatore per contesto trading
    if symbol:
        return TradeLoggerAdapter(logger, {'symbol': symbol})
    
    return logger


class TradeLoggerAdapter(logging.LoggerAdapter):
    """Adapter per aggiungere contesto ai log"""
    
    def process(self, msg: str, kwargs: dict) -> tuple:
        # Aggiungi simbolo al messaggio
        symbol = self.extra.get('symbol', '')
        if symbol:
            msg = f"[{symbol}] {msg}"
        return msg, kwargs
    
    def trade(self, side: str, price: float, size: float, **kwargs) -> None:
        """Log specializzato per trade"""
        emoji = "🟢" if side == "LONG" else "🔴"
        self.info(f"{emoji} {side} @ ${price:.2f} | Size: ${size:.2f}", **kwargs)
    
    def signal(self, score: float, action: str, **kwargs) -> None:
        """Log specializzato per segnali"""
        self.debug(f"📊 Score: {score:+.2f} → {action}", **kwargs)
    
    def pnl(self, pnl_value: float, pnl_pct: float, **kwargs) -> None:
        """Log specializzato per P&L"""
        emoji = "💰" if pnl_value >= 0 else "💸"
        color = "profit" if pnl_value >= 0 else "loss"
        self.info(f"{emoji} P&L: ${pnl_value:+.2f} ({pnl_pct:+.2%})", **kwargs)


# === Funzioni helper per compatibilità ===

def log_info(msg: str, symbol: str = None) -> None:
    """Helper per logging info (retrocompatibilità)"""
    logger = get_logger('bot', symbol)
    logger.info(msg)

def log_warning(msg: str, symbol: str = None) -> None:
    """Helper per logging warning"""
    logger = get_logger('bot', symbol)
    logger.warning(msg)

def log_error(msg: str, symbol: str = None, exc_info: bool = False) -> None:
    """Helper per logging error"""
    logger = get_logger('bot', symbol)
    logger.error(msg, exc_info=exc_info)

def log_trade(side: str, price: float, size: float, symbol: str = None) -> None:
    """Helper per logging trade"""
    logger = get_logger('bot', symbol)
    if isinstance(logger, TradeLoggerAdapter):
        logger.trade(side, price, size)
    else:
        emoji = "🟢" if side == "LONG" else "🔴"
        logger.info(f"[{symbol}] {emoji} {side} @ ${price:.2f} | Size: ${size:.2f}")


# Test del modulo
if __name__ == "__main__":
    setup_logging(level="DEBUG")
    
    logger = get_logger(__name__, symbol="BTCUSDT")
    
    logger.debug("Questo è un messaggio DEBUG")
    logger.info("Questo è un messaggio INFO")
    logger.warning("Questo è un messaggio WARNING")
    logger.error("Questo è un messaggio ERROR")
    
    # Test metodi specializzati
    if isinstance(logger, TradeLoggerAdapter):
        logger.trade("LONG", 45000.50, 100.0)
        logger.signal(4.5, "OPEN LONG")
        logger.pnl(150.25, 0.015)
    
    print("\n✅ Logger test completato! Controlla logs/")
