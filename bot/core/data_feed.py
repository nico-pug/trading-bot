"""
DataFeed - Gestione dati di mercato con caching, error handling e parallelizzazione

Miglioramenti rispetto alla versione originale:
- Caching TTL per evitare ricalcoli inutili
- Retry con backoff esponenziale per errori API
- Rate limiting per rispettare limiti Binance
- Parallelizzazione fetch dati (DOM, derivati, L/S ratio)
- Type hints completi
- Logging professionale

Usage:
    from bot.core import DataFeed
    
    feed = DataFeed()
    feed.find_symbol('BTC')
    df = feed.fetch_ohlcv_advanced()
    deriv, dom, ls = feed.fetch_all_parallel()
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import time
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from threading import Lock

# Import logging e config
try:
    from bot.utils.logger import get_logger
    from bot.utils.config import load_config
except ImportError:
    # Fallback se usato standalone
    import logging
    def get_logger(name, symbol=None):
        return logging.getLogger(name)
    def load_config():
        return None


# === DECORATORI ===

def retry_on_error(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Decoratore per retry con backoff esponenziale.
    Gestisce errori API Binance in modo robusto.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(__name__)
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, 
                        ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Tentativo {attempt + 1}/{max_retries} fallito: {e}. Retry in {delay:.1f}s")
                    time.sleep(delay)
                except ccxt.RateLimitExceeded as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** (attempt + 2)), max_delay)
                    logger.warning(f"Rate limit raggiunto. Attendo {delay:.1f}s")
                    time.sleep(delay)
                except Exception as e:
                    # Errori non recuperabili
                    logger.error(f"Errore non recuperabile in {func.__name__}: {e}")
                    raise
            
            logger.error(f"Max retry ({max_retries}) superati per {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


class TTLCache:
    """
    Cache semplice con Time-To-Live.
    Thread-safe per uso con parallelizzazione.
    """
    
    def __init__(self, ttl_seconds: int = 60):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Ottiene valore dalla cache se non scaduto"""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return value
                else:
                    del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Imposta valore in cache"""
        with self._lock:
            self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Svuota la cache"""
        with self._lock:
            self._cache.clear()


class DataFeed:
    """
    Gestore dati di mercato robusto.
    
    Features:
    - Connessione a Binance Futures
    - Indicatori tecnici avanzati
    - Pattern price action (Order Blocks, FVG, MSS)
    - Dati derivati (funding, OI, basis)
    - DOM e Long/Short ratio
    - Caching intelligente
    - Error handling robusto
    """
    
    def __init__(self, enable_cache: bool = True, cache_ttl: int = 30):
        """
        Inizializza DataFeed.
        
        Args:
            enable_cache: Abilita caching dei dati
            cache_ttl: Secondi di validità cache
        """
        self.logger = get_logger(__name__)
        self.config = load_config()
        
        # Inizializza exchange
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }
        
        # Applica config se disponibile
        if self.config:
            api_cfg = self.config.api
            exchange_config['timeout'] = api_cfg.timeout * 1000  # ms
            exchange_config['rateLimit'] = 1200 // (api_cfg.rate_limit_requests // 60)
        
        self.exchange = ccxt.binanceusdm(exchange_config)
        
        self.symbol: Optional[str] = None
        self.btc_symbol = 'BTC/USDT:USDT'
        
        # Caching
        self._cache_enabled = enable_cache
        self._ohlcv_cache = TTLCache(ttl_seconds=cache_ttl)
        self._deriv_cache = TTLCache(ttl_seconds=cache_ttl)
        self._dom_cache = TTLCache(ttl_seconds=cache_ttl // 2)  # DOM più frequente
        self._ls_cache = TTLCache(ttl_seconds=cache_ttl * 2)  # L/S ratio meno frequente
        
        self.logger.info("DataFeed inizializzato")
    
    @retry_on_error(max_retries=3)
    def find_symbol(self, user_input: str) -> bool:
        """
        Cerca il simbolo corretto su Binance Futures.
        
        Args:
            user_input: Input utente (es. 'BTC', 'btc', 'BTCUSDT')
            
        Returns:
            True se trovato, False altrimenti
        """
        self.logger.info(f"Ricerca simbolo per: {user_input}")
        
        self.exchange.load_markets()
        
        target_base = user_input.upper().replace('/', '').replace('USDT', '')
        
        candidates = []
        for market_id, market in self.exchange.markets.items():
            if (market.get('swap') and 
                market.get('linear') and 
                market.get('base') == target_base and 
                market.get('quote') == 'USDT'):
                candidates.append(market['symbol'])
        
        if not candidates:
            self.logger.error(f"Nessun contratto Perpetual USDT trovato per '{target_base}'")
            return False
        
        self.symbol = candidates[0]
        self.logger.info(f"Mercato trovato: {self.symbol}")
        return True
    
    @retry_on_error(max_retries=3)
    def fetch_ohlcv_advanced(self, limit: int = 200, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Scarica OHLCV con indicatori tecnici avanzati.
        
        Args:
            limit: Numero candele da scaricare
            use_cache: Se usare cache
            
        Returns:
            DataFrame con OHLCV + indicatori, o None in caso di errore
        """
        if self.symbol is None:
            self.logger.error("Simbolo non impostato")
            return None
        
        # Check cache
        cache_key = f"{self.symbol}_{limit}"
        if use_cache and self._cache_enabled:
            cached = self._ohlcv_cache.get(cache_key)
            if cached is not None:
                self.logger.debug("OHLCV da cache")
                return cached
        
        # Determina timeframe da config
        timeframe = '15m'
        if self.config:
            timeframe = self.config.backtest.timeframe
        
        bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # === INDICATORI TECNICI ===
        
        # VWAP
        df.ta.vwap(append=True)
        
        # TWAP
        df['TWAP'] = df['close'].expanding().mean()
        
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        
        # RSI
        df['RSI'] = df.ta.rsi(length=14)
        
        # ATR
        df['ATR'] = df.ta.atr(length=14)
        
        # CVD (Cumulative Volume Delta)
        df['delta_vol'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        df['CVD'] = df['delta_vol'].cumsum()
        
        # Fibonacci Levels
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        range_20 = high_20 - low_20
        df['fib_0618'] = high_20 - (range_20 * 0.618)
        df['fib_0786'] = high_20 - (range_20 * 0.786)
        df['fib_0382'] = high_20 - (range_20 * 0.382)
        
        # Salva in cache
        if self._cache_enabled:
            self._ohlcv_cache.set(cache_key, df)
        
        self.logger.debug(f"OHLCV scaricato: {len(df)} candele")
        return df
    
    def identify_order_blocks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifica Order Blocks (zone di accumulo istituzionale).
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            Lista di Order Blocks recenti
        """
        order_blocks = []
        if df is None or len(df) < 10:
            return order_blocks
        
        for i in range(3, len(df) - 1):
            # Bullish OB: candela rossa seguita da forte movimento up
            if df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                if df['close'].iloc[i] > df['high'].iloc[i-1]:
                    order_blocks.append({
                        'type': 'BULLISH_OB',
                        'high': df['high'].iloc[i-1],
                        'low': df['low'].iloc[i-1],
                        'idx': i
                    })
            
            # Bearish OB: candela verde seguita da forte movimento down
            if df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                if df['close'].iloc[i] < df['low'].iloc[i-1]:
                    order_blocks.append({
                        'type': 'BEARISH_OB',
                        'high': df['high'].iloc[i-1],
                        'low': df['low'].iloc[i-1],
                        'idx': i
                    })
        
        return order_blocks[-5:] if order_blocks else []
    
    def identify_fvg(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifica Fair Value Gaps (imbalances).
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            Lista di FVG recenti
        """
        fvgs = []
        if df is None or len(df) < 5:
            return fvgs
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                fvgs.append({
                    'type': 'BULLISH_FVG',
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i-2],
                    'idx': i
                })
            
            # Bearish FVG
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                fvgs.append({
                    'type': 'BEARISH_FVG',
                    'top': df['low'].iloc[i-2],
                    'bottom': df['high'].iloc[i],
                    'idx': i
                })
        
        return fvgs[-5:] if fvgs else []
    
    def detect_liquidity_grab(self, df: pd.DataFrame) -> Optional[str]:
        """
        Rileva Liquidity Grabs / Sweeps.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            'BULLISH_SWEEP', 'BEARISH_SWEEP' o None
        """
        if df is None or len(df) < 20:
            return None
        
        current = df.iloc[-1]
        prev_high = df['high'].iloc[-20:-1].max()
        prev_low = df['low'].iloc[-20:-1].min()
        
        if current['high'] > prev_high and current['close'] < prev_high:
            return 'BEARISH_SWEEP'
        if current['low'] < prev_low and current['close'] > prev_low:
            return 'BULLISH_SWEEP'
        
        return None
    
    def detect_mss(self, df: pd.DataFrame) -> Optional[str]:
        """
        Rileva Market Structure Shift (break of structure).
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            'BULLISH_MSS', 'BEARISH_MSS' o None
        """
        if df is None or len(df) < 10:
            return None
        
        highs = df['high'].iloc[-10:]
        lows = df['low'].iloc[-10:]
        
        recent_high = highs.max()
        recent_low = lows.min()
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        if prev_close < recent_high and current_close > recent_high:
            return 'BULLISH_MSS'
        if prev_close > recent_low and current_close < recent_low:
            return 'BEARISH_MSS'
        
        return None
    
    @retry_on_error(max_retries=2)
    def fetch_derivatives_data(self, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Recupera dati derivati: OI, Funding, Basis.
        
        Returns:
            Dict con price, funding_rate, open_interest, basis_pct
        """
        if self.symbol is None:
            return None
        
        # Check cache
        if use_cache and self._cache_enabled:
            cached = self._deriv_cache.get(self.symbol)
            if cached is not None:
                return cached
        
        ticker = self.exchange.fetch_ticker(self.symbol)
        funding = ticker.get('info', {}).get('lastFundingRate', 0)
        
        # Open Interest
        open_interest = 0.0
        try:
            oi_data = self.exchange.fetch_open_interest(self.symbol)
            open_interest = float(oi_data.get('openInterestAmount', 0))
        except Exception:
            pass
        
        # Basis Spot-Futures
        basis = 0.0
        try:
            spot_symbol = self.symbol.replace(':USDT', '')
            spot_ticker = self.exchange.fetch_ticker(spot_symbol)
            if spot_ticker and spot_ticker.get('last'):
                basis = ((ticker['last'] - spot_ticker['last']) / spot_ticker['last']) * 100
        except Exception:
            pass
        
        result = {
            'price': ticker['last'],
            'funding_rate': float(funding),
            'open_interest': open_interest,
            'basis_pct': basis
        }
        
        if self._cache_enabled:
            self._deriv_cache.set(self.symbol, result)
        
        return result
    
    @retry_on_error(max_retries=2)
    def fetch_long_short_ratio(self, use_cache: bool = True) -> Dict[str, float]:
        """
        Recupera Long/Short Ratio da Binance API pubblica.
        
        Returns:
            Dict con long_short_ratio, long_account, short_account
        """
        default = {'long_short_ratio': 1.0, 'long_account': 0.5, 'short_account': 0.5}
        
        if self.symbol is None:
            return default
        
        # Check cache
        cache_key = f"ls_{self.symbol}"
        if use_cache and self._cache_enabled:
            cached = self._ls_cache.get(cache_key)
            if cached is not None:
                return cached
        
        symbol_clean = self.symbol.replace('/', '').replace(':USDT', '')
        url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol_clean}&period=15m&limit=1"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                result = {
                    'long_short_ratio': float(data[0]['longShortRatio']),
                    'long_account': float(data[0]['longAccount']),
                    'short_account': float(data[0]['shortAccount'])
                }
                if self._cache_enabled:
                    self._ls_cache.set(cache_key, result)
                return result
        
        return default
    
    @retry_on_error(max_retries=2)
    def fetch_order_book_depth(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Recupera Depth of Market (DOM).
        
        Returns:
            Dict con imbalance, bid_vol, ask_vol, key_bids, key_asks
        """
        default = {'imbalance': 0, 'bid_vol': 0, 'ask_vol': 0, 'key_bids': [], 'key_asks': []}
        
        if self.symbol is None:
            return default
        
        # Check cache
        cache_key = f"dom_{self.symbol}"
        if use_cache and self._cache_enabled:
            cached = self._dom_cache.get(cache_key)
            if cached is not None:
                return cached
        
        book = self.exchange.fetch_order_book(self.symbol, limit=20)
        bids = book['bids']
        asks = book['asks']
        
        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        
        key_bid_levels = sorted(bids, key=lambda x: x[1], reverse=True)[:3]
        key_ask_levels = sorted(asks, key=lambda x: x[1], reverse=True)[:3]
        
        result = {
            'imbalance': bid_vol - ask_vol,
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'key_bids': [b[0] for b in key_bid_levels],
            'key_asks': [a[0] for a in key_ask_levels]
        }
        
        if self._cache_enabled:
            self._dom_cache.set(cache_key, result)
        
        return result
    
    def fetch_all_parallel(self) -> Tuple[Optional[Dict], Dict, Dict]:
        """
        Fetch parallelo di tutti i dati ausiliari.
        Migliora performance rispetto a chiamate sequenziali.
        
        Returns:
            Tuple (derivatives_data, dom_data, ls_ratio_data)
        """
        results = {
            'deriv': None,
            'dom': {'imbalance': 0, 'bid_vol': 0, 'ask_vol': 0, 'key_bids': [], 'key_asks': []},
            'ls': {'long_short_ratio': 1.0, 'long_account': 0.5, 'short_account': 0.5}
        }
        
        def fetch_deriv():
            return ('deriv', self.fetch_derivatives_data())
        
        def fetch_dom():
            return ('dom', self.fetch_order_book_depth())
        
        def fetch_ls():
            return ('ls', self.fetch_long_short_ratio())
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(fetch_deriv),
                executor.submit(fetch_dom),
                executor.submit(fetch_ls)
            ]
            
            for future in as_completed(futures):
                try:
                    key, value = future.result()
                    if value is not None:
                        results[key] = value
                except Exception as e:
                    self.logger.warning(f"Errore in fetch parallelo: {e}")
        
        return results['deriv'], results['dom'], results['ls']
    
    def clear_cache(self) -> None:
        """Svuota tutte le cache"""
        self._ohlcv_cache.clear()
        self._deriv_cache.clear()
        self._dom_cache.clear()
        self._ls_cache.clear()
        self.logger.debug("Cache svuotata")


# === TEST ===
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from bot.utils.logger import setup_logging
    setup_logging(level="INFO")
    
    feed = DataFeed()
    
    if feed.find_symbol('BTC'):
        print(f"\n[OK] Simbolo: {feed.symbol}")
        
        # Test OHLCV
        df = feed.fetch_ohlcv_advanced(limit=50)
        if df is not None:
            print(f"[OK] OHLCV: {len(df)} candele")
            print(f"     Ultimo prezzo: ${df['close'].iloc[-1]:,.2f}")
        
        # Test fetch parallelo
        print("\n[TEST] Fetch parallelo...")
        import time
        start = time.time()
        deriv, dom, ls = feed.fetch_all_parallel()
        elapsed = time.time() - start
        print(f"[OK] Fetch parallelo completato in {elapsed:.2f}s")
        
        if deriv:
            print(f"     Funding rate: {deriv['funding_rate']:.6f}")
        print(f"     DOM imbalance: {dom['imbalance']:.2f}")
        print(f"     L/S ratio: {ls['long_short_ratio']:.2f}")
        
        # Test cache
        print("\n[TEST] Cache...")
        start = time.time()
        df2 = feed.fetch_ohlcv_advanced(limit=50)
        elapsed = time.time() - start
        print(f"[OK] Seconda chiamata (cache): {elapsed:.3f}s")
        
        print("\n[OK] Tutti i test completati!")

