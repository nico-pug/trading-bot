"""
Base Scraper Interface

Implementa rate limiting e struttura comune.
"""

import time
import functools
from abc import ABC, abstractmethod
from typing import List, Dict, Any

def rate_limit(calls: int, period: int = 60):
    """
    Decorator per rate limiting semplice.
    Args:
        calls: Numero max chiamate
        period: Periodo in secondi
    """
    def decorator(func):
        history = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal history
            now = time.time()
            
            # Pulisci history vecchia
            history = [t for t in history if now - t < period]
            
            if len(history) >= calls:
                sleep_time = period - (now - history[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                history = [t for t in history if time.time() - t < period]
            
            history.append(time.time())
            return func(*args, **kwargs)
        return wrapper
    return decorator

class BaseScraper(ABC):
    """Interfaccia base per scraper"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def scrape(self, symbol: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Scarica dati per un simbolo.
        Returns:
            List[Dict]: [{'text': '...', 'url': '...', 'timestamp': ..., 'score': ...}]
        """
        pass
