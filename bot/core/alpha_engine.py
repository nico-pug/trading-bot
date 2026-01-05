"""
AlphaEngine - Generazione segnali di trading

Miglioramenti rispetto alla versione originale:
- EMA smoothing dello score per stabilità
- Normalizzazione cross-asset
- Weightings configurabili
- Logging dettagliato dei segnali
- Type hints completi

Usage:
    from bot.core import AlphaEngine, DataFeed
    
    feed = DataFeed()
    alpha = AlphaEngine()
    
    df = feed.fetch_ohlcv_advanced()
    deriv, dom, ls = feed.fetch_all_parallel()
    
    score, signals = alpha.analyze(df, deriv, dom, ls, feed)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from collections import deque

# Import logging e config
try:
    from bot.utils.logger import get_logger
    from bot.utils.config import load_config
except ImportError:
    import logging
    def get_logger(name, symbol=None):
        return logging.getLogger(name)
    def load_config():
        return None


class AlphaEngine:
    """
    Motore di generazione segnali multi-fattore.
    
    Analizza:
    - Indicatori tecnici (VWAP, BB, RSI, Fibonacci)
    - Price action (Order Blocks, FVG, MSS, Sweeps)
    - Dati derivati (Funding, OI, Basis)
    - Sentiment di mercato (Long/Short ratio, DOM)
    - On-chain (SOPR, NUPL, MVRV) se disponibili
    
    Include EMA smoothing per stabilizzare lo score.
    """
    
    # Volatilità base per normalizzazione cross-asset
    ASSET_VOLATILITY = {
        'BTC': 1.0,      # Baseline
        'ETH': 1.2,
        'SOL': 1.8,
        'DOGE': 2.0,
        'XRP': 1.5,
        'BNB': 1.3,
        'ADA': 1.6,
        'LTC': 1.4,
        'DEFAULT': 1.5
    }
    
    def __init__(self, 
                 ema_period: int = 5, 
                 enable_smoothing: bool = True,
                 enable_normalization: bool = True):
        """
        Inizializza AlphaEngine.
        
        Args:
            ema_period: Periodo per EMA smoothing dello score
            enable_smoothing: Se abilitare EMA smoothing
            enable_normalization: Se abilitare normalizzazione cross-asset
        """
        self.logger = get_logger(__name__)
        self.config = load_config()
        
        # EMA smoothing
        self._ema_period = ema_period
        self._enable_smoothing = enable_smoothing
        self._score_history: deque = deque(maxlen=ema_period * 2)
        self._ema_multiplier = 2 / (ema_period + 1)
        self._last_ema: Optional[float] = None
        
        # Normalizzazione
        self._enable_normalization = enable_normalization
        self._current_asset: Optional[str] = None
        
        # Tracciamento pattern
        self.last_ob = None
        self.last_fvg = None
        
        self.logger.info(f"AlphaEngine inizializzato (EMA: {ema_period}, smoothing: {enable_smoothing})")
    
    def set_asset(self, symbol: str) -> None:
        """
        Imposta l'asset corrente per normalizzazione.
        
        Args:
            symbol: Es. 'BTC/USDT:USDT' o 'BTCUSDT'
        """
        # Estrai base asset
        base = symbol.upper().split('/')[0].replace('USDT', '')
        self._current_asset = base
        self.logger.debug(f"Asset impostato: {base}")
    
    def _get_volatility_factor(self) -> float:
        """Ottiene fattore di volatilità per normalizzazione"""
        if not self._enable_normalization or self._current_asset is None:
            return 1.0
        return self.ASSET_VOLATILITY.get(self._current_asset, self.ASSET_VOLATILITY['DEFAULT'])
    
    def _apply_ema_smoothing(self, raw_score: float) -> float:
        """
        Applica EMA smoothing allo score per stabilizzarlo.
        
        Args:
            raw_score: Score grezzo
            
        Returns:
            Score smussato con EMA
        """
        if not self._enable_smoothing:
            return raw_score
        
        self._score_history.append(raw_score)
        
        if self._last_ema is None:
            # Prima osservazione: usa SMA
            if len(self._score_history) >= self._ema_period:
                self._last_ema = sum(list(self._score_history)[-self._ema_period:]) / self._ema_period
            else:
                self._last_ema = raw_score
        else:
            # EMA = (Score - EMA_prev) * multiplier + EMA_prev
            self._last_ema = (raw_score - self._last_ema) * self._ema_multiplier + self._last_ema
        
        return self._last_ema
    
    def analyze(self, 
                df: Optional[pd.DataFrame], 
                deriv: Optional[Dict[str, Any]], 
                dom: Optional[Dict[str, Any]], 
                ls_ratio: Optional[Dict[str, float]], 
                feed: Any,
                onchain: Optional[Dict[str, float]] = None) -> Tuple[float, List[str]]:
        """
        Analisi multi-fattore completa.
        
        Args:
            df: DataFrame OHLCV con indicatori
            deriv: Dati derivati (funding, OI, basis)
            dom: Depth of Market
            ls_ratio: Long/Short ratio
            feed: DataFeed per pattern detection
            onchain: Dati on-chain opzionali
            
        Returns:
            Tuple (score_normalizzato, lista_segnali)
        """
        raw_score = 0.0
        signals: List[str] = []
        
        if df is None or len(df) < 20:
            return 0.0, []
        
        current = df.iloc[-1]
        vol_factor = self._get_volatility_factor()
        
        # === VWAP (peso 1.0) ===
        vwap_col = [c for c in df.columns if 'VWAP' in c]
        if vwap_col:
            vwap_val = current.get(vwap_col[0])
            if pd.notna(vwap_val):
                if current['close'] > vwap_val:
                    raw_score += 1.0
                    signals.append("ABOVE_VWAP")
                else:
                    raw_score -= 1.0
                    signals.append("BELOW_VWAP")
        
        # === TWAP (peso 0.5) ===
        twap = current.get('TWAP')
        if pd.notna(twap):
            if current['close'] > twap:
                raw_score += 0.5
        
        # === Bollinger Bands (peso 1.5) ===
        bbu_col = [c for c in df.columns if 'BBU' in c]
        bbl_col = [c for c in df.columns if 'BBL' in c]
        if bbu_col and bbl_col:
            bbu = current.get(bbu_col[0])
            bbl = current.get(bbl_col[0])
            if pd.notna(bbu) and pd.notna(bbl):
                if current['close'] > bbu:
                    raw_score -= 1.5
                    signals.append("BB_OVERBOUGHT")
                elif current['close'] < bbl:
                    raw_score += 1.5
                    signals.append("BB_OVERSOLD")
        
        # === Fibonacci Golden Pocket (peso 1.0) ===
        fib_618 = current.get('fib_0618')
        fib_382 = current.get('fib_0382')
        if pd.notna(fib_618) and pd.notna(fib_382):
            if fib_618 < current['close'] < fib_382:
                raw_score += 1.0
                signals.append("IN_GOLDEN_POCKET")
        
        # === CVD Divergence (peso 1.0) ===
        if len(df) > 5 and 'CVD' in df.columns:
            price_trend = current['close'] - df['close'].iloc[-5]
            cvd_trend = current['CVD'] - df['CVD'].iloc[-5]
            if price_trend > 0 and cvd_trend < 0:
                raw_score -= 1.0
                signals.append("CVD_BEARISH_DIV")
            elif price_trend < 0 and cvd_trend > 0:
                raw_score += 1.0
                signals.append("CVD_BULLISH_DIV")
        
        # === RSI Extremes (peso 1.0) ===
        rsi = current.get('RSI')
        if pd.notna(rsi):
            if rsi < 30:
                raw_score += 1.0
                signals.append("RSI_OVERSOLD")
            elif rsi > 70:
                raw_score -= 1.0
                signals.append("RSI_OVERBOUGHT")
        
        # === Funding Rate (peso 2.0) ===
        if deriv:
            fr = deriv.get('funding_rate', 0)
            if fr < -0.0001:
                raw_score += 2.0
                signals.append("NEG_FUNDING")
            elif fr > 0.001:
                raw_score -= 2.0
                signals.append("HIGH_FUNDING")
            
            # Basis
            basis = deriv.get('basis_pct', 0)
            if basis > 0.5:
                raw_score -= 0.5
                signals.append("FUTURES_PREMIUM")
            elif basis < -0.5:
                raw_score += 0.5
                signals.append("FUTURES_DISCOUNT")
        
        # === Long/Short Ratio (peso 1.5) ===
        if ls_ratio:
            ratio = ls_ratio.get('long_short_ratio', 1.0)
            if ratio > 2.0:
                raw_score -= 1.5
                signals.append("CROWD_LONG")
            elif ratio < 0.5:
                raw_score += 1.5
                signals.append("CROWD_SHORT")
        
        # === DOM Imbalance (peso 0.5) ===
        if dom:
            imbalance = dom.get('imbalance', 0)
            if imbalance > 0:
                raw_score += 0.5
            else:
                raw_score -= 0.5
        
        # === Order Blocks (peso 1.0) ===
        if hasattr(feed, 'identify_order_blocks'):
            obs = feed.identify_order_blocks(df)
            if obs:
                last_ob = obs[-1]
                if last_ob['type'] == 'BULLISH_OB' and current['close'] > last_ob['low']:
                    raw_score += 1.0
                    signals.append("BULLISH_OB")
                elif last_ob['type'] == 'BEARISH_OB' and current['close'] < last_ob['high']:
                    raw_score -= 1.0
                    signals.append("BEARISH_OB")
        
        # === Fair Value Gaps (peso 0.5) ===
        if hasattr(feed, 'identify_fvg'):
            fvgs = feed.identify_fvg(df)
            if fvgs:
                last_fvg = fvgs[-1]
                if last_fvg['type'] == 'BULLISH_FVG':
                    raw_score += 0.5
                    signals.append("BULLISH_FVG")
                elif last_fvg['type'] == 'BEARISH_FVG':
                    raw_score -= 0.5
                    signals.append("BEARISH_FVG")
        
        # === Liquidity Sweep (peso 2.0) ===
        if hasattr(feed, 'detect_liquidity_grab'):
            sweep = feed.detect_liquidity_grab(df)
            if sweep:
                if sweep == 'BULLISH_SWEEP':
                    raw_score += 2.0
                    signals.append("BULLISH_SWEEP")
                elif sweep == 'BEARISH_SWEEP':
                    raw_score -= 2.0
                    signals.append("BEARISH_SWEEP")
        
        # === Market Structure Shift (peso 1.5) ===
        if hasattr(feed, 'detect_mss'):
            mss = feed.detect_mss(df)
            if mss:
                if mss == 'BULLISH_MSS':
                    raw_score += 1.5
                    signals.append("BULLISH_MSS")
                elif mss == 'BEARISH_MSS':
                    raw_score -= 1.5
                    signals.append("BEARISH_MSS")
        
        # === On-Chain (peso 0.5 ciascuno) ===
        if onchain:
            sopr = onchain.get('sopr', 1.0)
            nupl = onchain.get('nupl', 0.0)
            mvrv = onchain.get('mvrv', 1.0)
            
            if sopr > 1.05:
                raw_score -= 0.5
                signals.append("HIGH_SOPR")
            elif sopr < 0.95:
                raw_score += 0.5
                signals.append("LOW_SOPR")
            
            if nupl > 0.3:
                raw_score -= 0.5
                signals.append("NUPL_GREED")
            elif nupl < -0.3:
                raw_score += 0.5
                signals.append("NUPL_FEAR")
            
            if mvrv > 1.5:
                raw_score -= 0.5
                signals.append("MVRV_HIGH")
            elif mvrv < 0.8:
                raw_score += 0.5
                signals.append("MVRV_LOW")
        
        # === NORMALIZZAZIONE CROSS-ASSET ===
        # Asset più volatili richiedono score più alto
        normalized_score = raw_score / vol_factor
        
        # === EMA SMOOTHING ===
        smoothed_score = self._apply_ema_smoothing(normalized_score)
        
        self.logger.debug(f"Score: raw={raw_score:.2f}, norm={normalized_score:.2f}, smooth={smoothed_score:.2f}")
        
        return smoothed_score, signals
    
    def get_signal_strength(self, score: float) -> str:
        """
        Converte score numerico in forza del segnale.
        
        Args:
            score: Score calcolato
            
        Returns:
            'STRONG_LONG', 'LONG', 'NEUTRAL', 'SHORT', 'STRONG_SHORT'
        """
        if score > 5:
            return "STRONG_LONG"
        elif score > 3:
            return "LONG"
        elif score < -5:
            return "STRONG_SHORT"
        elif score < -3:
            return "SHORT"
        return "NEUTRAL"
    
    def reset(self) -> None:
        """Reset dello stato interno (per nuovo asset o backtest)"""
        self._score_history.clear()
        self._last_ema = None
        self.last_ob = None
        self.last_fvg = None


# === TEST ===
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from bot.utils.logger import setup_logging
    setup_logging(level="DEBUG")
    
    # Test standalone
    alpha = AlphaEngine(ema_period=3, enable_smoothing=True)
    alpha.set_asset('BTC')
    
    # Simula alcuni score
    scores = [2.5, 3.0, 2.8, 4.5, 3.2, -1.0, -2.5]
    
    print("\n=== TEST EMA SMOOTHING ===")
    for raw in scores:
        smoothed = alpha._apply_ema_smoothing(raw)
        print(f"Raw: {raw:+.2f} -> Smoothed: {smoothed:+.2f}")
    
    print("\n=== TEST VOLATILITY FACTORS ===")
    for asset in ['BTC', 'SOL', 'DOGE', 'ETH']:
        alpha.set_asset(asset)
        factor = alpha._get_volatility_factor()
        print(f"{asset}: volatility factor = {factor}")
    
    print("\n✅ AlphaEngine test completato!")
