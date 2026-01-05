"""
ExecutionEngine - Esecuzione trade con trailing stop e multi-posizione

Miglioramenti rispetto alla versione originale:
- Trailing stop dinamico
- Supporto multi-posizione (max configurabile)
- Fee e slippage integrati
- Position ID univoci
- Logging dettagliato

Usage:
    from bot.core import ExecutionEngine, RiskManager
    
    risk = RiskManager()
    exec = ExecutionEngine(risk_manager=risk)
    
    pos_id = exec.open_trade('LONG', price=45000, atr=500, score=4.5, signals=['RSI_OVERSOLD'])
    exec.manage_positions(current_price=46000)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

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
class Position:
    """Rappresenta una posizione aperta"""
    id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    size: float
    sl: float  # Stop Loss
    tp: float  # Take Profit
    trailing_sl: Optional[float] = None  # Trailing Stop (se attivo)
    trailing_pct: float = 0.0  # Percentuale trailing
    score: float = 0.0
    signals: List[str] = field(default_factory=list)
    opened_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_long(self) -> bool:
        return self.side == 'LONG'
    
    @property
    def current_sl(self) -> float:
        """Ritorna trailing SL se attivo, altrimenti SL fisso"""
        if self.trailing_sl is not None:
            return self.trailing_sl
        return self.sl


class ExecutionEngine:
    """
    Motore di esecuzione trade.
    
    Features:
    - Trailing stop dinamico
    - Multi-posizione (configurabile)
    - Fee e slippage integrati via RiskManager
    - Position tracking con ID univoci
    """
    
    def __init__(self,
                 initial_balance: float = 10000.0,
                 risk_manager = None,
                 perf_tracker = None,
                 max_positions: int = 3,
                 enable_trailing: bool = True,
                 trailing_activation_pct: float = 0.015,  # Attiva trailing dopo +1.5%
                 trailing_distance_pct: float = 0.01):    # Trail a 1% dal max
        """
        Inizializza ExecutionEngine.
        
        Args:
            initial_balance: Capitale iniziale
            risk_manager: Istanza RiskManager
            perf_tracker: Istanza PerformanceTracker (opzionale)
            max_positions: Numero massimo posizioni aperte
            enable_trailing: Abilita trailing stop
            trailing_activation_pct: % profit per attivare trailing
            trailing_distance_pct: Distanza trailing dal prezzo massimo
        """
        self.logger = get_logger(__name__)
        self.config = load_config()
        
        self.balance = initial_balance
        self.risk_manager = risk_manager
        self.perf_tracker = perf_tracker
        
        self.max_positions = max_positions
        self.enable_trailing = enable_trailing
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct
        
        self.positions: List[Position] = []
        
        # Tracking
        self._last_score = 0.0
        self._last_signals: List[str] = []
        self._closed_trades: List[Dict] = []
        
        # Config override
        if self.config:
            self.sl_mult = self.config.trading.sl_multiplier
            self.tp_mult = self.config.trading.tp_multiplier
        else:
            self.sl_mult = 2.0
            self.tp_mult = 3.0
        
        self.logger.info(f"ExecutionEngine inizializzato: max_pos={max_positions}, trailing={enable_trailing}")
    
    def set_symbol_config(self, symbol: str) -> None:
        """Carica configurazione specifica per simbolo"""
        if self.config:
            sym_cfg = self.config.get_symbol_config(symbol)
            self.sl_mult = sym_cfg.sl_multiplier
            self.tp_mult = sym_cfg.tp_multiplier
            self.logger.debug(f"Config {symbol}: SL={self.sl_mult}x, TP={self.tp_mult}x")
    
    def can_open_position(self) -> tuple:
        """
        Verifica se è possibile aprire una nuova posizione.
        
        Returns:
            Tuple (can_open, reason)
        """
        # Check max positions
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) raggiunte"
        
        # Check risk manager
        if self.risk_manager:
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                return False, reason
        
        return True, "OK"
    
    def open_trade(self,
                   side: str,
                   price: float,
                   atr: float,
                   score: float,
                   signals: List[str],
                   symbol: str = "UNKNOWN",
                   use_kelly: bool = False) -> Optional[str]:
        """
        Apre una nuova posizione.
        
        Args:
            side: 'LONG' o 'SHORT'
            price: Prezzo di entrata
            atr: ATR corrente per calcolo SL/TP
            score: Score del segnale
            signals: Lista segnali attivi
            symbol: Simbolo trading
            use_kelly: Se usare Kelly per position sizing
            
        Returns:
            Position ID se aperto, None altrimenti
        """
        if any(p.symbol == symbol for p in self.positions):
            self.logger.warning(f"Posizione già aperta su {symbol}. Ignoro nuovo segnale.")
            return None

        can_open, reason = self.can_open_position()
        if not can_open:
            self.logger.warning(f"Impossibile aprire trade: {reason}")
            return None
        
        self._last_score = score
        self._last_signals = signals
        
        # Position sizing
        if use_kelly and self.risk_manager:
            kelly_pct = self.risk_manager.calculate_kelly()
            risk_pct = min(kelly_pct, 0.02)  # Cap a 2%
        else:
            risk_pct = self.risk_manager.risk_per_trade if self.risk_manager else 0.01
        
        # Calcola SL e TP
        if side == 'LONG':
            sl = price - (atr * self.sl_mult)
            tp = price + (atr * self.tp_mult)
        else:
            sl = price + (atr * self.sl_mult)
            tp = price - (atr * self.tp_mult)
        
        # Calcola size
        risk_amount = self.balance * risk_pct
        sl_distance = abs(price - sl)
        if sl_distance == 0:
            self.logger.error("SL distance = 0, impossibile calcolare size")
            return None
        
        size = risk_amount / (sl_distance / price)  # In quote currency
        
        # Limite leva (5x)
        max_size = self.balance * 5
        if size > max_size:
            size = max_size
            self.logger.warning(f"Size limitata a {max_size:.2f} (5x leva)")
        
        # Crea posizione
        pos_id = str(uuid.uuid4())[:8]
        position = Position(
            id=pos_id,
            symbol=symbol,
            side=side,
            entry_price=price,
            size=size,
            sl=sl,
            tp=tp,
            trailing_pct=self.trailing_distance_pct,
            score=score,
            signals=signals.copy()
        )
        
        self.positions.append(position)
        
        self.logger.info(f"[OPEN] #{pos_id} {side} @ {price:.2f} | Size: ${size:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        self.logger.debug(f"        Score: {score:.1f} | Signals: {signals[:5]}")
        
        return pos_id
    
    def manage_positions(self, current_price: float, symbol: str = "UNKNOWN") -> List[Dict]:
        """
        Gestisce tutte le posizioni aperte.
        Verifica SL/TP e aggiorna trailing stop.
        
        Args:
            current_price: Prezzo corrente
            symbol: Simbolo per filtrare posizioni
            
        Returns:
            Lista di trade chiusi
        """
        closed = []
        
        for pos in self.positions[:]:  # Copia per iterare
            if pos.symbol != symbol and symbol != "UNKNOWN":
                continue
            
            close = False
            exit_price = current_price
            exit_reason = ""
            
            # Update trailing stop
            if self.enable_trailing:
                self._update_trailing_stop(pos, current_price)
            
            # Check chiusura
            if pos.is_long:
                if current_price <= pos.current_sl:
                    close = True
                    exit_price = pos.current_sl
                    exit_reason = "SL" if pos.trailing_sl is None else "TRAILING_SL"
                elif current_price >= pos.tp:
                    close = True
                    exit_price = pos.tp
                    exit_reason = "TP"
            else:  # SHORT
                if current_price >= pos.current_sl:
                    close = True
                    exit_price = pos.current_sl
                    exit_reason = "SL" if pos.trailing_sl is None else "TRAILING_SL"
                elif current_price <= pos.tp:
                    close = True
                    exit_price = pos.tp
                    exit_reason = "TP"
            
            if close:
                trade_result = self._close_position(pos, exit_price, exit_reason)
                closed.append(trade_result)
        
        return closed
    
    def _update_trailing_stop(self, pos: Position, current_price: float) -> None:
        """Aggiorna trailing stop per una posizione"""
        if pos.is_long:
            # Profit %
            profit_pct = (current_price - pos.entry_price) / pos.entry_price
            
            if profit_pct >= self.trailing_activation_pct:
                # Trailing attivo
                new_trail = current_price * (1 - pos.trailing_pct)
                
                if pos.trailing_sl is None or new_trail > pos.trailing_sl:
                    old_sl = pos.trailing_sl or pos.sl
                    pos.trailing_sl = new_trail
                    if pos.trailing_sl > old_sl:
                        self.logger.debug(f"#{pos.id} Trailing SL: {old_sl:.2f} -> {new_trail:.2f}")
        else:
            # SHORT
            profit_pct = (pos.entry_price - current_price) / pos.entry_price
            
            if profit_pct >= self.trailing_activation_pct:
                new_trail = current_price * (1 + pos.trailing_pct)
                
                if pos.trailing_sl is None or new_trail < pos.trailing_sl:
                    old_sl = pos.trailing_sl or pos.sl
                    pos.trailing_sl = new_trail
                    if pos.trailing_sl < old_sl:
                        self.logger.debug(f"#{pos.id} Trailing SL: {old_sl:.2f} -> {new_trail:.2f}")
    
    def _close_position(self, pos: Position, exit_price: float, reason: str) -> Dict:
        """Chiude una posizione e registra il risultato"""
        # Calcola P&L
        if self.risk_manager:
            gross_pnl, fees, slip, net_pnl = self.risk_manager.calculate_pnl_with_costs(
                pos.entry_price, exit_price, pos.size, pos.side
            )
            self.risk_manager.record_trade(
                pos.symbol, pos.side, pos.entry_price, exit_price, pos.size
            )
        else:
            # Calcolo semplificato senza risk manager
            if pos.is_long:
                gross_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size
            else:
                gross_pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size
            fees, slip, net_pnl = 0, 0, gross_pnl
        
        self.balance += net_pnl
        pnl_pct = net_pnl / pos.size * 100 if pos.size > 0 else 0
        
        result_type = "PROFIT" if net_pnl > 0 else "LOSS"
        self.logger.info(f"[{result_type}] #{pos.id} {reason} @ {exit_price:.2f} | "
                        f"Net: ${net_pnl:+.2f} ({pnl_pct:+.1f}%) | Balance: ${self.balance:.2f}")
        
        trade_result = {
            'id': pos.id,
            'symbol': pos.symbol,
            'side': pos.side,
            'entry': pos.entry_price,
            'exit': exit_price,
            'size': pos.size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'score': pos.score,
            'signals': pos.signals,
            'duration': (datetime.now() - pos.opened_at).seconds
        }
        
        self._closed_trades.append(trade_result)
        self.positions.remove(pos)
        
        return trade_result
    
    def close_all_positions(self, current_price: float, reason: str = "MANUAL") -> List[Dict]:
        """Chiude tutte le posizioni aperte"""
        closed = []
        for pos in self.positions[:]:
            result = self._close_position(pos, current_price, reason)
            closed.append(result)
        return closed
    
    def get_open_positions(self) -> List[Dict]:
        """Ritorna lista posizioni aperte come dict"""
        return [
            {
                'id': p.id,
                'symbol': p.symbol,
                'side': p.side,
                'entry': p.entry_price,
                'size': p.size,
                'sl': p.current_sl,
                'tp': p.tp,
                'trailing_active': p.trailing_sl is not None,
                'score': p.score
            }
            for p in self.positions
        ]
    
    def get_stats(self) -> Dict:
        """Statistiche esecuzione"""
        if not self._closed_trades:
            return {'total_trades': 0}
        
        wins = [t for t in self._closed_trades if t['net_pnl'] > 0]
        losses = [t for t in self._closed_trades if t['net_pnl'] <= 0]
        
        return {
            'total_trades': len(self._closed_trades),
            'open_positions': len(self.positions),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self._closed_trades) if self._closed_trades else 0,
            'total_pnl': sum(t['net_pnl'] for t in self._closed_trades),
            'balance': self.balance,
            'trailing_closes': len([t for t in self._closed_trades if 'TRAILING' in t['reason']])
        }


# === TEST ===
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from bot.utils.logger import setup_logging
    from bot.core.risk_manager import RiskManager
    
    setup_logging(level="INFO")
    
    rm = RiskManager(initial_capital=10000)
    exec_engine = ExecutionEngine(
        initial_balance=10000,
        risk_manager=rm,
        max_positions=2,
        enable_trailing=True,
        trailing_activation_pct=0.01,  # 1% per test
        trailing_distance_pct=0.005    # 0.5%
    )
    
    print("\n=== EXECUTION ENGINE TEST ===")
    
    # Test apertura trade
    print("\n[TEST] Apertura trade:")
    pos_id = exec_engine.open_trade(
        side='LONG',
        price=100.0,
        atr=2.0,
        score=4.5,
        signals=['RSI_OVERSOLD', 'BULLISH_OB'],
        symbol='BTCUSDT'
    )
    print(f"  Posizione aperta: {pos_id}")
    
    # Simula movimento prezzo (profit)
    print("\n[TEST] Simulazione trailing stop:")
    prices = [100, 101, 102, 103, 104, 103.5, 103]  # Sale poi ritraccia
    
    for price in prices:
        pos = exec_engine.positions[0] if exec_engine.positions else None
        if pos:
            profit_pct = (price - pos.entry_price) / pos.entry_price * 100
            trail = pos.trailing_sl if pos.trailing_sl else "N/A"
            print(f"  Price: ${price:.2f} | Profit: {profit_pct:+.1f}% | Trail SL: {trail}")
        
        closed = exec_engine.manage_positions(price, 'BTCUSDT')
        if closed:
            print(f"  -> Trade chiuso: {closed[0]['reason']}")
            break
    
    # Stats
    print("\n[TEST] Stats:")
    stats = exec_engine.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n[OK] Test completato!")
