"""
PerformanceTracker - Trade Journal e metriche avanzate

Usage:
    from bot.core import PerformanceTracker
    
    tracker = PerformanceTracker(journal_file="trades.csv")
    tracker.log_trade(...)
"""

import os
import csv
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import logging e config
try:
    from bot.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name, symbol=None):
        return logging.getLogger(name)


class PerformanceTracker:
    """
    Trade Journal e calcolo metriche avanzate.
    
    Features:
    - CSV trade journal
    - Alpha/Beta calculation
    - Treynor Ratio
    - Jensen's Alpha
    """
    
    def __init__(self, journal_file: str = "trade_journal.csv"):
        self.logger = get_logger(__name__)
        self.journal_file = journal_file
        self.trades: List[Dict] = []
        self._init_journal()
    
    def _init_journal(self) -> None:
        """Inizializza il Trade Journal CSV"""
        # Ensure directory exists
        folder = os.path.dirname(self.journal_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        if not os.path.exists(self.journal_file):
            with open(self.journal_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'entry_price', 'exit_price',
                    'size', 'gross_pnl', 'fees', 'slippage', 'net_pnl', 
                    'pnl_pct', 'score', 'signals', 'reason', 'notes'
                ])
            self.logger.info(f"Trade journal creato: {self.journal_file}")
    
    def log_trade(self,
                  symbol: str,
                  side: str,
                  entry: float,
                  exit_price: float,
                  size: float,
                  gross_pnl: float = 0,
                  fees: float = 0,
                  slippage: float = 0,
                  net_pnl: float = 0,
                  score: float = 0,
                  signals: List[str] = None,
                  reason: str = "",
                  notes: str = "") -> None:
        """Registra trade nel journal"""
        pnl_pct = (net_pnl / size) * 100 if size > 0 else 0
        
        with open(self.journal_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                side,
                f"{entry:.4f}",
                f"{exit_price:.4f}",
                f"{size:.4f}",
                f"{gross_pnl:.2f}",
                f"{fees:.4f}",
                f"{slippage:.4f}",
                f"{net_pnl:.2f}",
                f"{pnl_pct:.2f}%",
                f"{score:.1f}",
                str(signals or []),
                reason,
                notes
            ])
        
        self.trades.append({
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'side': side,
            'symbol': symbol
        })
    
    def calculate_alpha(self, portfolio_return: float, benchmark_return: float) -> float:
        """Alpha - rendimento superiore al benchmark"""
        return portfolio_return - benchmark_return
    
    def calculate_beta(self, 
                       asset_returns: List[float], 
                       benchmark_returns: List[float]) -> float:
        """Beta - correlazione rispetto al mercato"""
        if not asset_returns or not benchmark_returns:
            return 1.0
        if len(asset_returns) < 10 or len(benchmark_returns) < 10:
            return 1.0
        
        min_len = min(len(asset_returns), len(benchmark_returns))
        asset_arr = np.array(asset_returns[-min_len:])
        bench_arr = np.array(benchmark_returns[-min_len:])
        
        covariance = np.cov(asset_arr, bench_arr)[0, 1]
        variance = np.var(bench_arr)
        
        if variance == 0:
            return 1.0
        return covariance / variance
    
    def calculate_treynor_ratio(self, 
                                 portfolio_return: float, 
                                 beta: float, 
                                 risk_free: float = 0.02) -> float:
        """Treynor Ratio - performance per unità di rischio sistematico"""
        if beta == 0:
            return 0.0
        return (portfolio_return - risk_free) / beta
    
    def calculate_jensens_alpha(self,
                                 portfolio_return: float,
                                 benchmark_return: float,
                                 beta: float,
                                 risk_free: float = 0.02) -> float:
        """Jensen's Alpha - abilità gestionale"""
        expected_return = risk_free + beta * (benchmark_return - risk_free)
        return portfolio_return - expected_return
    
    def get_summary(self) -> Dict:
        """Sommario performance"""
        if not self.trades:
            return {'total_trades': 0}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades),
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'avg_pnl': sum(t['pnl'] for t in self.trades) / len(self.trades),
            'best_trade': max(t['pnl'] for t in self.trades),
            'worst_trade': min(t['pnl'] for t in self.trades)
        }
