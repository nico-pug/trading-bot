"""
Trading Bot Core Class
Assembla tutti i componenti (DataFeed, Alpha, Risk, Execution, ML, Sentiment).
"""

import time
import signal
import sys
from typing import Optional
from bot.utils.logger import get_logger
from bot.utils.config import load_config
from bot.core.data_feed import DataFeed
from bot.core.alpha_engine import AlphaEngine
from bot.core.risk_manager import RiskManager
from bot.core.execution import ExecutionEngine
from bot.core.performance import PerformanceTracker
try:
    from bot.ml.filter import MLFilter
except ImportError:
    MLFilter = None
try:
    from bot.sentiment.analyzer import SentimentAnalyzer
except ImportError:
    SentimentAnalyzer = None

class TradingBot:
    def __init__(self, symbol: str, strategy: str = 'standard'):
        self.symbol = symbol
        self.logger = get_logger(f"Bot_{symbol.replace('/','-')}")
        self.config = load_config()
        self.running = False
        
        # 1. Data Feed
        self.feed = DataFeed()
        if not self.feed.find_symbol(symbol):
            self.logger.error(f"Simbolo {symbol} non trovato su Binance Futures!")
            # Fallback or raise, but for now logic continues or subsequent calls fail
            # Raising exception to stop faulty bot start
            raise ValueError(f"Symbol {symbol} not found or invalid for Binance Futures")
        
        # 2. Risk & Performance
        self.risk_manager = RiskManager()
        self.perf_tracker = PerformanceTracker(journal_file=f"data/journals/journal_{symbol.replace('/','').replace(':','_')}.csv")
        
        # 3. Execution Engine
        self.execution = ExecutionEngine(
            risk_manager=self.risk_manager,
            perf_tracker=self.perf_tracker
        )
        self.execution.set_symbol_config(symbol)
        
        # 4. Alpha Engine
        self.alpha = AlphaEngine()
        self.alpha.set_asset(symbol)
        
        # 5. ML Filter
        self.ml_filter = None
        if MLFilter and self.config and self.config.ml.enabled:
            try:
                self.ml_filter = MLFilter(symbol)
                self.logger.info(f"ML Filter attivato per {symbol}")
            except Exception as e:
                self.logger.warning(f"ML Filter non disponibile: {e}")

        # 6. Sentiment Analyzer
        self.sentiment = None
        if SentimentAnalyzer: # Configurabile in futuro
            try:
                self.sentiment = SentimentAnalyzer()
                self.logger.info("Sentiment Analyzer attivato")
            except Exception as e:
                self.logger.warning(f"Sentiment Analyzer errore: {e}")
                
        # Register signal handlers
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
    def start(self):
        """Avvia il loop di trading"""
        self.running = True
        self.logger.info(f"Bot avviato su {self.symbol}")
        
        while self.running:
            try:
                start_time = time.time()
                self._tick()
                
                # Smart Sleep: sleep only for the remainder of the interval
                elapsed = time.time() - start_time
                interval = self.config.system.scan_interval if self.config else 60
                sleep_time = max(0.0, interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"Tick cycle took too long ({elapsed:.2f}s > {interval}s)")
            except Exception as e:
                self.logger.error(f"Errore nel loop principale: {e}", exc_info=True)
                time.sleep(5) # Reduced backoff in case of error
                
    def stop(self, signum=None, frame=None):
        """Ferma il bot gentilmente"""
        # Close any resources if needed
        self.logger.info("Arresto in corso...")
        self.running = False
        
    def _tick(self):
        """Ciclo di analisi singolo"""
        # 1. Scarica dati
        df = self.feed.fetch_ohlcv_advanced(limit=100)
        deriv, dom, ls_ratio = self.feed.fetch_all_parallel()
        
        if df is None:
            self.logger.warning("Dati OHLCV non disponibili")
            return
            
        current_price = df['close'].iloc[-1]
        
        # 2. Gestione posizioni aperte (SL/TP, Trailing)
        self.execution.manage_positions(current_price, self.symbol)
        
        # 3. Analisi Alpha
        score, signals = self.alpha.analyze(df, deriv, dom, ls_ratio, self.feed)
        
        # 4. Sentiment Integration (se disponibile)
        if self.sentiment:
            try:
                sent_res = self.sentiment.analyze_symbol(self.symbol.split('/')[0])
                # Modifica score in base al sentiment
                if sent_res['score'] > 0.5:
                    score += 1.0
                    signals.append("SENTIMENT_BULLISH")
                elif sent_res['score'] < -0.5:
                    score -= 1.0
                    signals.append("SENTIMENT_BEARISH")
            except Exception as e:
                self.logger.error(f"Errore sentiment: {e}")
                
        # 5. ML Filtering (se attivo)
        if self.ml_filter:
            try:
                # Check for potential movement
                # We check logic for both sides or biased by current score
                idx = len(df) - 1
                
                # Boost if ML confirms current bias
                if score > 0:
                    prob = self.ml_filter.check_signal(df, idx, score, 'LONG')
                    if prob > 0.6:
                        score += 0.5
                        signals.append(f"ML_LONG_CONFIRM({prob:.2f})")
                elif score < 0:
                    prob = self.ml_filter.check_signal(df, idx, score, 'SHORT')
                    if prob > 0.6:
                        score -= 0.5
                        signals.append(f"ML_SHORT_CONFIRM({prob:.2f})")
            except Exception as e:
                self.logger.warning(f"ML check error: {e}")
            
        # 6. Decisione Trading
        thresh_long = self.execution.config.get_symbol_config(self.symbol).score_threshold_long if hasattr(self.execution, 'config') and self.execution.config else 3.0
        thresh_short = self.execution.config.get_symbol_config(self.symbol).score_threshold_short if hasattr(self.execution, 'config') and self.execution.config else -3.0
        
        atr = df['ATR'].iloc[-1] if 'ATR' in df else current_price * 0.01
        
        if score >= thresh_long:
            self.execution.open_trade('LONG', current_price, atr, score, signals, self.symbol)
        elif score <= thresh_short:
            self.execution.open_trade('SHORT', current_price, atr, score, signals, self.symbol)
            
        self.logger.info(f"{self.symbol} | Price: {current_price:.2f} | Score: {score:.1f} | Active Signal: {len(signals)}\n{'-'*50}")

if __name__ == "__main__":
    # Test standalone
    bot = TradingBot('BTC/USDT:USDT')
    print("Bot inizializzato. Start test...")
    # bot.start() # Commentato per evitare loop infinito
