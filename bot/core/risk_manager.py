"""
RiskManager - Gestione rischio avanzata

Miglioramenti rispetto alla versione originale:
- Fee e slippage nei calcoli P&L
- Cooldown dopo consecutive losses o drawdown elevato
- Metriche avanzate (Kelly, Sharpe, Sortino, Risk of Ruin)
- Type hints e logging

Usage:
    from bot.core import RiskManager
    
    risk = RiskManager(initial_capital=10000)
    kelly_size = risk.calculate_kelly()
    can_trade = risk.can_trade()
"""

import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

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


@dataclass
class TradeResult:
    """Risultato di un trade"""
    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    pnl_pct: float
    

class RiskManager:
    """
    Gestore del rischio con calcoli avanzati.
    
    Features:
    - Kelly Criterion per position sizing
    - Expected Value tracking
    - Drawdown monitoring
    - Risk of Ruin calculation
    - Sharpe & Sortino ratios
    - Fee & slippage integration
    - Cooldown after losses
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 win_rate: float = 0.55,
                 rr_ratio: float = 1.5,
                 risk_per_trade: float = 0.01,
                 max_drawdown_limit: float = 0.20,
                 fee_rate: float = 0.0004,
                 slippage: float = 0.0005,
                 cooldown_losses: int = 3,
                 cooldown_minutes: int = 60):
        """
        Inizializza RiskManager.
        
        Args:
            initial_capital: Capitale iniziale
            win_rate: Win rate stimato del sistema
            rr_ratio: Risk/Reward ratio medio
            risk_per_trade: Percentuale rischio per trade (1% = 0.01)
            max_drawdown_limit: Limite drawdown per stop trading
            fee_rate: Fee taker (0.04% = 0.0004)
            slippage: Slippage medio stimato
            cooldown_losses: Numero consecutive losses prima del cooldown
            cooldown_minutes: Minuti di cooldown
        """
        self.logger = get_logger(__name__)
        self.config = load_config()
        
        # Capitale
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.current_capital = initial_capital
        
        # Parametri
        self.win_rate = win_rate
        self.rr_ratio = rr_ratio
        self.risk_per_trade = risk_per_trade
        self.max_drawdown_limit = max_drawdown_limit
        
        # Costi
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Carica da config se disponibile
        if self.config:
            self.fee_rate = self.config.trading.fee_rate
            self.slippage = self.config.trading.slippage
            self.max_drawdown_limit = self.config.trading.max_drawdown_limit
            self.risk_per_trade = self.config.trading.risk_per_trade
        
        # Cooldown
        self._cooldown_losses = cooldown_losses
        self._cooldown_minutes = cooldown_minutes
        self._consecutive_losses = 0
        self._cooldown_until: Optional[datetime] = None
        
        # Storico
        self.returns: deque = deque(maxlen=100)
        self.trade_history: List[TradeResult] = []
        
        # Statistiche
        self._total_trades = 0
        self._winning_trades = 0
        self._total_fees = 0.0
        self._total_slippage = 0.0
        
        self.logger.info(f"RiskManager inizializzato: capital=${initial_capital:,.2f}, fee={fee_rate:.4%}, slippage={slippage:.4%}")
    
    # === CALCOLI P&L CON COSTI ===
    
    def calculate_pnl_with_costs(self, 
                                  entry_price: float, 
                                  exit_price: float, 
                                  size: float, 
                                  side: str) -> Tuple[float, float, float, float]:
        """
        Calcola P&L includendo fee e slippage.
        
        Args:
            entry_price: Prezzo di entrata
            exit_price: Prezzo di uscita
            size: Size della posizione (in quote currency)
            side: 'LONG' o 'SHORT'
            
        Returns:
            Tuple (gross_pnl, fees, slippage_cost, net_pnl)
        """
        # P&L lordo
        if side == 'LONG':
            gross_pnl = (exit_price - entry_price) / entry_price * size
        else:
            gross_pnl = (entry_price - exit_price) / entry_price * size
        
        # Fee (entry + exit)
        fees = size * self.fee_rate * 2
        
        # Slippage (entry + exit)
        slippage_cost = size * self.slippage * 2
        
        # P&L netto
        net_pnl = gross_pnl - fees - slippage_cost
        
        return gross_pnl, fees, slippage_cost, net_pnl
    
    def calculate_realistic_size(self,
                                  price: float,
                                  atr: float,
                                  sl_multiplier: float = 2.0) -> float:
        """
        Calcola size realistico considerando fee e slippage.
        
        Args:
            price: Prezzo corrente
            atr: ATR per calcolo distanza SL
            sl_multiplier: Moltiplicatore ATR per SL
            
        Returns:
            Size ottimale
        """
        # Distanza SL
        sl_distance = atr * sl_multiplier
        sl_pct = sl_distance / price
        
        # Rischio effettivo (include costi)
        effective_risk = self.risk_per_trade
        total_costs = (self.fee_rate + self.slippage) * 2
        
        # Rischio disponibile dopo costi
        available_risk = effective_risk - total_costs
        if available_risk <= 0:
            self.logger.warning("Costi superano il rischio per trade!")
            return 0.0
        
        # Size basata su rischio disponibile
        risk_amount = self.current_capital * available_risk
        size = risk_amount / sl_pct
        
        return size
    
    # === KELLY CRITERION ===
    
    def calculate_kelly(self, 
                        win_rate: Optional[float] = None, 
                        rr_ratio: Optional[float] = None) -> float:
        """
        Kelly Criterion per position sizing ottimale.
        Usa Half-Kelly per essere conservativi.
        
        Returns:
            Percentuale del capitale da rischiare (0-0.25)
        """
        wr = win_rate or self._calculate_actual_win_rate() or self.win_rate
        rr = rr_ratio or self._calculate_actual_rr() or self.rr_ratio
        
        # Kelly = W - [(1-W) / R]
        kelly = wr - ((1 - wr) / rr)
        
        # Half-Kelly per conservatività
        half_kelly = kelly / 2
        
        # Cap tra 0 e 25%
        return max(0, min(half_kelly, 0.25))
    
    def _calculate_actual_win_rate(self) -> Optional[float]:
        """Calcola win rate effettivo dagli ultimi trade"""
        if self._total_trades < 10:
            return None
        return self._winning_trades / self._total_trades
    
    def _calculate_actual_rr(self) -> Optional[float]:
        """Calcola R/R effettivo dagli ultimi trade"""
        if len(self.trade_history) < 10:
            return None
        
        wins = [t.net_pnl for t in self.trade_history if t.net_pnl > 0]
        losses = [abs(t.net_pnl) for t in self.trade_history if t.net_pnl < 0]
        
        if not wins or not losses:
            return None
        
        return np.mean(wins) / np.mean(losses)
    
    # === EXPECTED VALUE ===
    
    def calculate_expected_value(self, 
                                  win_rate: Optional[float] = None,
                                  avg_win: float = 100,
                                  avg_loss: float = 66) -> float:
        """
        Expected Value del sistema.
        EV = (Win% * Avg Win) - (Loss% * Avg Loss)
        """
        wr = win_rate or self.win_rate
        return (wr * avg_win) - ((1 - wr) * avg_loss)
    
    # === DRAWDOWN ===
    
    def update_drawdown(self, current_capital: Optional[float] = None) -> float:
        """
        Aggiorna e calcola drawdown corrente.
        
        Returns:
            Drawdown come percentuale (0-1)
        """
        if current_capital is not None:
            self.current_capital = current_capital
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        if self.peak_capital == 0:
            return 0.0
        
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return drawdown
    
    def is_max_drawdown_exceeded(self) -> bool:
        """Verifica se drawdown ha superato il limite"""
        dd = self.update_drawdown()
        exceeded = dd >= self.max_drawdown_limit
        if exceeded:
            self.logger.warning(f"Max drawdown raggiunto: {dd:.1%} >= {self.max_drawdown_limit:.1%}")
        return exceeded
    
    # === RISK OF RUIN ===
    
    def check_risk_of_ruin(self, 
                           win_rate: Optional[float] = None,
                           risk_per_trade: Optional[float] = None) -> float:
        """
        Calcola Risk of Ruin.
        RoR = ((1-edge)/(1+edge))^units
        
        Returns:
            Probabilità di rovina (0-1)
        """
        wr = win_rate or self.win_rate
        rpt = risk_per_trade or self.risk_per_trade
        
        # Edge del sistema
        edge = (wr * 2) - 1
        
        if edge <= 0:
            return 1.0  # 100% probabilità di rovina
        
        # Numero di "unità" di rischio
        units = 1 / rpt
        
        ror = ((1 - edge) / (1 + edge)) ** units
        return min(ror, 1.0)
    
    # === SHARPE & SORTINO ===
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe Ratio annualizzato.
        Assume 96 periodi da 15min al giorno.
        """
        if len(self.returns) < 10:
            return 0.0
        
        returns_arr = np.array(self.returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        
        if std_return == 0:
            return 0.0
        
        # Annualizzato
        periods_per_day = 96  # 15min candles
        annual_factor = np.sqrt(periods_per_day * 365)
        daily_rf = risk_free_rate / 365 / periods_per_day
        
        sharpe = (mean_return - daily_rf) / std_return * annual_factor
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sortino Ratio - considera solo downside volatility.
        """
        if len(self.returns) < 10:
            return 0.0
        
        returns_arr = np.array(self.returns)
        mean_return = np.mean(returns_arr)
        downside_returns = returns_arr[returns_arr < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        periods_per_day = 96
        annual_factor = np.sqrt(periods_per_day * 365)
        daily_rf = risk_free_rate / 365 / periods_per_day
        
        sortino = (mean_return - daily_rf) / downside_std * annual_factor
        return sortino
    
    # === COOLDOWN ===
    
    def _check_cooldown_trigger(self) -> None:
        """Verifica se attivare cooldown"""
        if self._consecutive_losses >= self._cooldown_losses:
            self._cooldown_until = datetime.now() + timedelta(minutes=self._cooldown_minutes)
            self.logger.warning(f"Cooldown attivato: {self._consecutive_losses} loss consecutive. "
                              f"Ripresa alle {self._cooldown_until.strftime('%H:%M')}")
    
    def is_in_cooldown(self) -> bool:
        """Verifica se in periodo di cooldown"""
        if self._cooldown_until is None:
            return False
        
        if datetime.now() >= self._cooldown_until:
            self._cooldown_until = None
            self._consecutive_losses = 0
            self.logger.info("Cooldown terminato, trading riabilitato")
            return False
        
        return True
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Verifica se è possibile aprire nuovi trade.
        
        Returns:
            Tuple (can_trade, reason_if_blocked)
        """
        # Check cooldown
        if self.is_in_cooldown():
            remaining = (self._cooldown_until - datetime.now()).seconds // 60
            return False, f"In cooldown per altri {remaining} minuti"
        
        # Check drawdown
        if self.is_max_drawdown_exceeded():
            return False, f"Max drawdown {self.max_drawdown_limit:.0%} superato"
        
        # Check capitale minimo
        if self.current_capital < self.initial_capital * 0.1:
            return False, "Capitale sceso sotto il 10% iniziale"
        
        return True, "OK"
    
    # === TRACKING ===
    
    def record_trade(self,
                     symbol: str,
                     side: str,
                     entry_price: float,
                     exit_price: float,
                     size: float) -> TradeResult:
        """
        Registra un trade completato.
        
        Returns:
            TradeResult con tutti i dettagli
        """
        gross_pnl, fees, slippage_cost, net_pnl = self.calculate_pnl_with_costs(
            entry_price, exit_price, size, side
        )
        
        pnl_pct = net_pnl / size * 100 if size > 0 else 0
        
        result = TradeResult(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            gross_pnl=gross_pnl,
            fees=fees,
            slippage=slippage_cost,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct
        )
        
        # Update state
        self.current_capital += net_pnl
        self.trade_history.append(result)
        self.returns.append(pnl_pct / 100)
        
        self._total_trades += 1
        self._total_fees += fees
        self._total_slippage += slippage_cost
        
        if net_pnl > 0:
            self._winning_trades += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._check_cooldown_trigger()
        
        self.update_drawdown()
        
        self.logger.info(f"Trade {side}: gross=${gross_pnl:+.2f}, fees=${fees:.2f}, "
                        f"slip=${slippage_cost:.2f}, net=${net_pnl:+.2f}")
        
        return result
    
    def add_return(self, pnl_pct: float) -> None:
        """Aggiunge return per calcoli statistici (retrocompatibilità)"""
        self.returns.append(pnl_pct)
    
    # === STATS ===
    
    def get_stats(self) -> Dict:
        """Ottiene statistiche complete"""
        dd = self.update_drawdown()
        can_trade, reason = self.can_trade()
        
        return {
            'capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_pnl': self.current_capital - self.initial_capital,
            'total_pnl_pct': (self.current_capital / self.initial_capital - 1) * 100,
            'drawdown': dd,
            'max_drawdown_limit': self.max_drawdown_limit,
            'total_trades': self._total_trades,
            'winning_trades': self._winning_trades,
            'win_rate': self._winning_trades / self._total_trades if self._total_trades > 0 else 0,
            'total_fees': self._total_fees,
            'total_slippage': self._total_slippage,
            'sharpe': self.calculate_sharpe_ratio(),
            'sortino': self.calculate_sortino_ratio(),
            'kelly': self.calculate_kelly(),
            'risk_of_ruin': self.check_risk_of_ruin(),
            'can_trade': can_trade,
            'trade_status': reason
        }


# === TEST ===
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from bot.utils.logger import setup_logging
    setup_logging(level="INFO")
    
    rm = RiskManager(initial_capital=10000, fee_rate=0.0004, slippage=0.0005)
    
    print("\n=== RISK MANAGER TEST ===")
    
    # Test calcolo P&L con costi
    print("\n[TEST] Calcolo P&L con costi:")
    gross, fees, slip, net = rm.calculate_pnl_with_costs(
        entry_price=100, exit_price=105, size=1000, side='LONG'
    )
    print(f"  LONG $100 -> $105, size $1000")
    print(f"  Gross P&L: ${gross:.2f}")
    print(f"  Fees: ${fees:.2f}")
    print(f"  Slippage: ${slip:.2f}")
    print(f"  Net P&L: ${net:.2f}")
    
    # Test Kelly
    print(f"\n[TEST] Kelly Criterion: {rm.calculate_kelly():.1%}")
    
    # Test trade recording
    print("\n[TEST] Recording trades:")
    
    # Trade vincente
    result = rm.record_trade('BTCUSDT', 'LONG', 90000, 91000, 1000)
    print(f"  Trade 1 (win): net=${result.net_pnl:+.2f}")
    
    # Trade perdente
    result = rm.record_trade('BTCUSDT', 'SHORT', 91000, 91500, 1000)
    print(f"  Trade 2 (loss): net=${result.net_pnl:+.2f}")
    
    # Stats
    print("\n[TEST] Stats finali:")
    stats = rm.get_stats()
    print(f"  Capital: ${stats['capital']:.2f}")
    print(f"  Total P&L: ${stats['total_pnl']:+.2f} ({stats['total_pnl_pct']:+.1f}%)")
    print(f"  Win rate: {stats['win_rate']:.0%}")
    print(f"  Total fees: ${stats['total_fees']:.2f}")
    print(f"  Can trade: {stats['can_trade']} ({stats['trade_status']})")
    
    print("\n[OK] Test completato!")
