"""
BacktestOptimizer - Ottimizzazione parametri tramite Grid Search parallela

Features:
- Grid Search parallela (multiprocessing)
- Caching risultati
- Fitness function configurabile
- Report migliori parametri

Usage:
    from bot.backtest.optimizer import BacktestOptimizer
    
    opt = BacktestOptimizer(symbol='BTC/USDT:USDT')
    best_params = opt.optimize(
        param_grid={'sl_mult': [1.5, 2.0, 2.5], 'tp_mult': [2.0, 3.0]},
        n_jobs=-1
    )
"""

import itertools
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Callable, Optional
import time
import sys
import os

# Fix path per esecuzione come script
sys.path.insert(0, os.getcwd())

# Imports
try:
    from bot.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name, symbol=None):
        return logging.getLogger(name)

# Imports deferred to avoid circular dependency
BacktestEngine = None
BacktestResult = None


class BacktestOptimizer:
    """Ottimizzatore parametri backtest"""
    
    def __init__(self, symbol: str, initial_capital: float = 10000):
        self.logger = get_logger(__name__)
        self.symbol = symbol
        self.initial_capital = initial_capital
        
    def optimize(self, 
                 param_grid: Dict[str, List[Any]],
                 fitness_func: Optional[Callable] = None,
                 n_jobs: int = -1) -> Dict:
        """
        Esegue Grid Search parallela.
        
        Args:
            param_grid: Dizionario liste parametri es: {'sl': [1,2], 'tp': [2,3]}
            fitness_func: Funzione custom (result -> float)
            n_jobs: Numero processi (-1 = tutti i core)
            
        Returns:
            Dict coi migliori parametri
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        # Lazy import to avoid circular dependency
        global BacktestEngine, BacktestResult
        if BacktestEngine is None:
            try:
                from bot.backtest.engine import BacktestEngine, BacktestResult
            except ImportError:
                 # Fallback for direct script execution without package context
                 pass

        self.logger.info(f"Avvio ottimizzazione: {len(combinations)} combinazioni, n_jobs={n_jobs}")
        
        # Download dati una volta sola (main process)
        if BacktestEngine:
            engine = BacktestEngine(symbol=self.symbol)
            engine.download_historical_data(days=180, use_cache=True)
        else:
            self.logger.error("BacktestEngine not available")
            return {}
        # Note: I dati vengono ricaricati in ogni processo se non passati esplicitamente,
        # ma grazie al caching su disco sarà veloce.
        
        results = []
        start_time = time.time()
        
        # Esecuzione parallela
        with ProcessPoolExecutor(max_workers=None if n_jobs == -1 else n_jobs) as executor:
            # Sottometti task
            future_to_params = {
                executor.submit(self._run_single_backtest, dict(zip(keys, combo))): dict(zip(keys, combo))
                for combo in combinations
            }
            
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    res = future.result()
                    
                    # Calcola fitness score
                    score = self._calculate_fitness(res) if not fitness_func else fitness_func(res)
                    
                    results.append({
                        'params': params,
                        'result': res,
                        'score': score
                    })
                    
                    if len(results) % 10 == 0:
                        self.logger.info(f"Completati {len(results)}/{len(combinations)}")
                        
                except Exception as e:
                    self.logger.error(f"Errore optim {params}: {str(e)}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Ottimizzazione completata in {elapsed:.2f}s")
        
        if not results:
            return {}
            
        # Trova migliore
        best = max(results, key=lambda x: x['score'])
        
        self.logger.info(f"Migliori params: {best['params']}")
        self.logger.info(f"Best Score: {best['score']:.4f}, P&L: {best['result'].total_pnl_pct:.2f}%")
        
        self._print_top_results(results)
        
        return best['params']

    def _run_single_backtest(self, params: Dict) -> BacktestResult:
        """Helper eseguito nel sottoprocesso"""
        # Re-import necessario nel sottoprocesso
        from bot.backtest.engine import BacktestEngine
        
        # In multiprocessing ogni worker deve istanziare il suo engine
        engine = BacktestEngine(symbol=self.symbol, initial_capital=self.initial_capital)
        
        # Carica da cache (disk)
        engine.download_historical_data(days=180, use_cache=True)
        
        # Run
        return engine.run_backtest(silent=True, **params)
    
    def _calculate_fitness(self, res: BacktestResult) -> float:
        """Default fitness function: Composite Score"""
        # Mix di P&L, Sharpe, Drawdown, Profit Factor
        
        # P&L component (bounded)
        pnl_score = np.log1p(max(0, res.total_pnl_pct))
        
        # Sharpe component
        sharpe_score = min(res.sharpe_ratio, 3.0)  # Cap a 3
        
        # Drawdown penalty (esponenziale)
        dd_penalty = np.exp(res.max_drawdown * 5) - 1
        
        # Win rate bonus
        wr_bonus = (res.win_rate - 50) / 100 if res.win_rate > 50 else 0
        
        score = pnl_score * 0.4 + sharpe_score * 0.3 - dd_penalty * 0.4 + wr_bonus * 0.2
        return max(0, score)

    def _print_top_results(self, results: List[Dict], top_n: int = 5):
        """Stampa migliori risultati"""
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"{'TOP {top_n} CONFIGURATIONS':^80}")
        print(f"{'='*80}")
        print(f"{'#':<4} {'SCORE':<8} {'P&L %':<10} {'DD %':<10} {'SHARPE':<8} {'PARAMS'}")
        print(f"{'-'*80}")
        
        for i, item in enumerate(sorted_results[:top_n], 1):
            r = item['result']
            params_str = str(item['params'])
            print(f"{i:<4} {item['score']:<8.4f} {r.total_pnl_pct:<10.2f} {r.max_drawdown:<10.1%} {r.sharpe_ratio:<8.2f} {params_str}")
        print(f"{'='*80}\n")


# === TEST ===
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from bot.utils.logger import setup_logging
    setup_logging(level="INFO")
    
    # Check if we are running in main process
    # Windows richiede if __name__ == "__main__" per multiprocessing
    opt = BacktestOptimizer(symbol='BTC/USDT:USDT')
    
    print("\n[TEST] Avvio ottimizzazione...")
    param_grid = {
        'sl_mult': [1.5, 2.0],
        'tp_mult': [2.0, 3.0]
    }
    
    best = opt.optimize(param_grid, n_jobs=2)
    print("\n[OK] Test completato!")
