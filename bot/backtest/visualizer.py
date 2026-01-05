"""
BacktestVisualizer - Visualizzazione grafica risultati backtest

Features:
- Grafico Equity Curve
- Grafico Drawdown
- Istogramma distribuzione rendimenti
- Heatmap mensile (opzionale)

Usage:
    from bot.backtest.visualizer import BacktestVisualizer
    
    vis = BacktestVisualizer()
    vis.plot_equity_curve(results)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Any
import os

# Import logging
try:
    from bot.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name, symbol=None):
        return logging.getLogger(name)


class BacktestVisualizer:
    """Visualizzatore grafici per backtest"""
    
    def __init__(self, save_dir: str = "reports/plots"):
        self.logger = get_logger(__name__)
        self.save_dir = save_dir
        
        # Crea directory se non esiste
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Stile grafici
        plt.style.use('bmh')
        self.colors = {
            'equity': '#2962FF',  # Blue
            'drawdown': '#D50000', # Red
            'profit': '#00C853',   # Green
            'loss': '#DD2C00'      # Red
        }
    
    def plot_results(self, result: Any, show: bool = True, save: bool = True) -> Optional[str]:
        """Genera dashboard completa dei risultati"""
        
        # Estrai dati
        equity_curve = pd.DataFrame(result.equity_curve)
        if equity_curve.empty:
            self.logger.warning("Nessun dato equity da visualizzare")
            return None
            
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
        
        # Setup subplot
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Equity Curve (Top wide)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity(ax1, equity_curve)
        
        # 2. Drawdown (Middle Left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(ax2, equity_curve)
        
        # 3. Monthly Returns (Middle Right) - Opzionale/Semplificato
        # Per ora usiamo distribuzione ritorni giornalieri approssimata
        ax3 = fig.add_subplot(gs[1, 1])
        # self._plot_monthly_returns(ax3, equity_curve) # Richiede logica complessa
        self._plot_rolling_volatility(ax3, equity_curve)
        
        # 4. Trade Distribution (Bottom Left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_trade_distribution(ax4, result.trades)
        
        # 5. Cumulative Returns vs Benchmark (Bottom Right - placeholder)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_win_loss_pie(ax5, result.trades)
        
        # Layout
        plt.tight_layout()
        
        # Save
        filename = None
        if save:
            filename = f"{self.save_dir}/backtest_report_{result.symbol.replace('/', '_').replace(':', '')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300)
            self.logger.info(f"Grafico salvato: {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return filename

    def _plot_equity(self, ax, equity_df):
        """Grafico curva dei profitti"""
        ax.plot(equity_df.index, equity_df['balance'], color=self.colors['equity'], linewidth=1.5)
        ax.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax.set_ylabel('Balance ($)')
        ax.grid(True, alpha=0.3)
        
        # Fill area
        ax.fill_between(equity_df.index, equity_df['balance'], equity_df['balance'].min(), alpha=0.1, color=self.colors['equity'])

    def _plot_drawdown(self, ax, equity_df):
        """Grafico drawdown sottomarino"""
        # Calcola drawdown
        peak = equity_df['balance'].cummax()
        drawdown = (equity_df['balance'] - peak) / peak
        
        ax.fill_between(equity_df.index, drawdown, 0, color=self.colors['drawdown'], alpha=0.3)
        ax.plot(equity_df.index, drawdown, color=self.colors['drawdown'], linewidth=1)
        ax.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

    def _plot_rolling_volatility(self, ax, equity_df, window=20):
        """Grafico volatilità rolling"""
        returns = equity_df['balance'].pct_change().dropna()
        if not returns.empty:
            vol = returns.rolling(window).std() * np.sqrt(252)
            ax.plot(returns.index, vol, color='#FF9800', linewidth=1.5)
            ax.set_title(f'Rolling Volatility ({window} periods)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')

    def _plot_trade_distribution(self, ax, trades):
        """Istogramma distribuzione P&L trade"""
        if not trades:
            ax.text(0.5, 0.5, 'No Trades', ha='center', va='center')
            return
            
        pnls = [t['pnl_pct'] for t in trades]
        
        # Color bars based on positive/negative
        n, bins, patches = ax.hist(pnls, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        for i in range(len(patches)):
            if bins[i] < 0:
                patches[i].set_facecolor(self.colors['loss'])
            else:
                patches[i].set_facecolor(self.colors['profit'])
                
        ax.set_title('Trade P&L Distribution (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('P&L %')
        ax.set_ylabel('Frequency')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)

    def _plot_win_loss_pie(self, ax, trades):
        """Pie chart win rate"""
        if not trades:
            ax.text(0.5, 0.5, 'No Trades', ha='center', va='center')
            return
            
        wins = len([t for t in trades if t['net_pnl'] > 0])
        losses = len([t for t in trades if t['net_pnl'] <= 0])
        
        # Se 0 trade, evita errore
        if wins + losses == 0:
            return
            
        ax.pie([wins, losses], labels=['Wins', 'Losses'], 
               colors=[self.colors['profit'], self.colors['loss']],
               autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
        ax.set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')


# === TEST ===
if __name__ == "__main__":
    # Mock result object for testing
    from dataclasses import dataclass
    
    @dataclass
    class MockResult:
        symbol: str = "BTC/TEST"
        trades: List = None
        equity_curve: List = None
        
    dates = pd.date_range(start="2025-01-01", periods=100)
    balance = [10000 * (1 + 0.01 * np.random.randn()) for _ in range(100)]
    balance = np.cumprod([1 + 0.001 * np.random.randn() for _ in range(100)]) * 10000
    
    equity_data = [{'timestamp': d, 'balance': b} for d, b in zip(dates, balance)]
    
    mock_trades = [
        {'pnl_pct': 1.5, 'net_pnl': 150},
        {'pnl_pct': -0.5, 'net_pnl': -50},
        {'pnl_pct': 2.0, 'net_pnl': 200},
        {'pnl_pct': -1.0, 'net_pnl': -100}
    ] * 5
    
    print("Test BacktestVisualizer...")
    vis = BacktestVisualizer()
    vis.plot_results(MockResult(trades=mock_trades, equity_curve=equity_data), save=True, show=False)
    print("Test completato.")
