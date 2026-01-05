"""
Unit Tests for AlphaEngine
"""
import unittest
import pandas as pd
import numpy as np
from bot.core.alpha_engine import AlphaEngine

class TestAlphaEngine(unittest.TestCase):
    def setUp(self):
        self.alpha = AlphaEngine(ema_period=3, enable_smoothing=True)
        self.alpha.set_asset('BTC')
        
    def _create_mock_df(self):
        # Create a simple DF
        dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
        df = pd.DataFrame(index=dates)
        df['open'] = 100.0
        df['high'] = 105.0
        df['low'] = 95.0
        df['close'] = 102.0
        df['volume'] = 1000
        df['VWAP'] = 101.0 # Close > VWAP = Bullish
        df['RSI'] = 50.0
        return df
        
    def test_smoothing(self):
        raw_scores = [1.0, 1.0, 1.0, -1.0, -1.0]
        smoothed = []
        for s in raw_scores:
            smoothed.append(self.alpha._apply_ema_smoothing(s))
            
        # First should be SMA (1/3, 2/3, 1), then EMA
        # Just check last value is negative but smoothed
        self.assertLess(smoothed[-1], 0)
        self.assertGreater(smoothed[-1], -1.0) # Not fully -1 due to lag
        
    def test_analyze_simple(self):
        df = self._create_mock_df()
        
        # Test basic bullish signal (Close > VWAP)
        score, signals = self.alpha.analyze(
            df, deriv=None, dom=None, ls_ratio=None, feed=None
        )
        
        # VWAP adds 1.0, normalized by volatility (BTC=1.0) -> 1.0
        # Smoothed over 3 period... initial might be exactly raw if history empty?
        # Check source code: if history < period, it returns raw (1st) or sma.
        # Let's check positive score
        self.assertGreater(score, 0)
        self.assertIn("ABOVE_VWAP", signals)
        
    def test_rsi_extremes(self):
        df = self._create_mock_df()
        df.loc[df.index[-1], 'RSI'] = 25 # Oversold -> Bullish
        
        score, signals = self.alpha.analyze(
            df, deriv=None, dom=None, ls_ratio=None, feed=None
        )
        self.assertIn("RSI_OVERSOLD", signals)
        self.assertGreater(score, 0)
        
    def test_volatility_normalization(self):
        self.alpha.set_asset('DOGE') # Vol factor 2.0
        df = self._create_mock_df()
        
        # Raw score 1.0 (VWAP)
        # Normalized = 1.0 / 2.0 = 0.5
        score, _ = self.alpha.analyze(
            df, deriv=None, dom=None, ls_ratio=None, feed=None
        )
        
        # Should be approx 0.5 (ignoring smoothing lag for first item)
        # AlphaEngine uses raw for first item if history is empty?
        # "if len(self._score_history) >= self._ema_period: ... else: self._last_ema = raw_score"
        # So first item is raw/normalized.
        self.assertAlmostEqual(score, 0.5)

if __name__ == '__main__':
    unittest.main()
