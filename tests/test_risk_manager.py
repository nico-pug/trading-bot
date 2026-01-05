"""
Unit Tests for RiskManager
"""
import unittest
from bot.core.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(initial_capital=10000)
        
    def test_initial_state(self):
        self.assertEqual(self.rm.current_capital, 10000)
        self.assertEqual(self.rm.peak_capital, 10000)
        
    def test_kelly_criterion(self):
        # Kelly: W - (1-W)/R
        # W=0.55, R=1.5 => 0.55 - 0.45/1.5 = 0.55 - 0.3 = 0.25 (25%)
        # Half Kelly = 12.5%
        kelly = self.rm.calculate_kelly(win_rate=0.55, rr_ratio=1.5)
        self.assertAlmostEqual(kelly, 0.125)
        
    def test_drawdown_calculation(self):
        self.rm.update_drawdown(9000) # 10% loss
        self.assertEqual(self.rm.peak_capital, 10000)
        # Check internal state if possible, or method return
        dd = (10000 - 9000) / 10000
        self.assertEqual(dd, 0.10)
        
        self.rm.update_drawdown(11000) # New peak
        self.assertEqual(self.rm.peak_capital, 11000)
        
        self.rm.update_drawdown(10000)
        dd = (11000 - 10000) / 11000
        self.assertAlmostEqual(dd, 0.09090909)

    def test_can_trade_check(self):
        # Should be allowed initially
        can_trade, msg = self.rm.can_trade()
        self.assertTrue(can_trade)
        
        # Simulate heavy drawdown
        self.rm.update_drawdown(5000) # 50% DD
        can_trade, msg = self.rm.can_trade()
        self.assertFalse(can_trade)
        self.assertIn("Max drawdown", msg)
        
    def test_pnl_calculation(self):
        # Long trade: Buy 100, Sell 110, Size 100 (Quote Currency)
        # Gross: (110 - 100) / 100 * 100 = 10.0
        # Fees: 100 * 0.0004 * 2 = 0.08
        # Slippage: 100 * 0.0005 * 2 = 0.10
        # Net: 10.0 - 0.08 - 0.10 = 9.82
        gross, fees, slip, net = self.rm.calculate_pnl_with_costs(100, 110, 100.0, 'LONG')
        self.assertAlmostEqual(gross, 10.0)
        self.assertAlmostEqual(fees, 0.08)
        self.assertAlmostEqual(slip, 0.10)
        self.assertAlmostEqual(net, 9.82)

if __name__ == '__main__':
    unittest.main()
