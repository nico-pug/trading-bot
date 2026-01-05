"""
Unit Tests for ExecutionEngine
"""
import unittest
from datetime import datetime
from bot.core.execution import ExecutionEngine, Position
from bot.core.risk_manager import RiskManager

class TestExecutionEngine(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(initial_capital=10000)
        self.exec = ExecutionEngine(initial_balance=10000, risk_manager=self.rm, max_positions=2)
        
    def test_open_trade(self):
        # Open LONG
        pos_id = self.exec.open_trade(
            side='LONG',
            price=100.0,
            atr=2.0,
            score=3.5,
            signals=['TEST'],
            symbol='BTC'
        )
        self.assertIsNotNone(pos_id)
        self.assertEqual(len(self.exec.positions), 1)
        self.assertEqual(self.exec.positions[0].id, pos_id)
        
        # Check SL/TP
        # Config default: SL 2x, TP 3x
        # SL = 100 - (2*2) = 96
        # TP = 100 + (2*3) = 106
        self.assertEqual(self.exec.positions[0].sl, 96.0)
        self.assertEqual(self.exec.positions[0].tp, 106.0)
        
    def test_max_positions_limit(self):
        self.exec.open_trade('LONG', 100, 1, 3, [], 'A')
        self.exec.open_trade('LONG', 100, 1, 3, [], 'B')
        
        # Third should fail
        pos_id = self.exec.open_trade('LONG', 100, 1, 3, [], 'C')
        self.assertIsNone(pos_id)
        self.assertEqual(len(self.exec.positions), 2)
        
    def test_trailing_stop(self):
        # Setup position
        self.exec.open_trade('LONG', 100.0, 2.0, 3.5, [], 'BTC')
        pos = self.exec.positions[0]
        
        # Price moves up +2% (102.0)
        # Activation pct default 1.5% -> Active
        # Trailing distance default 1% -> New SL = 102 * 0.99 = 100.98
        self.exec.manage_positions(102.0, 'BTC')
        
        self.assertIsNotNone(pos.trailing_sl)
        self.assertAlmostEqual(pos.trailing_sl, 102.0 * (1 - self.exec.trailing_distance_pct))
        self.assertGreater(pos.trailing_sl, pos.sl)
        
        # Price drops to 100.90 -> Should close
        closed = self.exec.manage_positions(100.90, 'BTC')
        self.assertTrue(len(closed) > 0)
        self.assertEqual(closed[0]['reason'], 'TRAILING_SL')
        self.assertEqual(len(self.exec.positions), 0)

    def test_close_position(self):
        self.exec.open_trade('LONG', 100, 2, 3, [], 'BTC')
        # Close manually via TP simulation
        closed = self.exec.manage_positions(110, 'BTC') # Hits TP (106)
        
        self.assertTrue(len(closed) > 0)
        self.assertEqual(closed[0]['reason'], 'TP')
        self.assertEqual(closed[0]['exit'], 106.0) # Should execute at TP price
        
        # Balance should increase
        # Entry 100, Exit 106. Gain 6 per unit.
        # Size logic: 1% risk (100$) / distance (4$) = 25 units
        # PnL = 25 * 6 = 150 - fees
        self.assertGreater(self.exec.balance, 10000)

if __name__ == '__main__':
    unittest.main()
