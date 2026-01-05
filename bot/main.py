"""
Trading Bot CLI Entry Point
Unifica tutte le funzioni (Trade, Backtest, Optimize, Multi-Bot).
"""

import argparse
import sys
import os
from bot.utils.logger import setup_logging

def cmd_trade(args):
    """Avvia bot singola istanza"""
    from bot.core.trader import TradingBot
    bot = TradingBot(args.symbol)
    bot.start()

def cmd_multi(args):
    """Avvia multi-bot manager"""
    from bot.core.manager import ProcessManager
    manager = ProcessManager()
    symbols = args.symbols
    if not symbols:
        print("Errore: specifica almeno un simbolo.")
        return
    manager.launch_multi(symbols)

def cmd_backtest(args):
    """Avvia backtest"""
    from bot.backtest.engine import BacktestEngine
    from bot.backtest.visualizer import BacktestVisualizer
    
    print(f"Avvio Backtest su {args.symbol} ({args.days} giorni)...")
    engine = BacktestEngine(args.symbol)
    engine.download_historical_data(days=args.days)
    metrics = engine.run_backtest()
    
    # Stampa report formattato
    engine.print_report(metrics)
        
    # Visualizzazione
    viz = BacktestVisualizer()
    viz.plot_results(metrics)
    print("\nGrafici salvati in reports/plots/")

def cmd_optimize(args):
    """Avvia ottimizzazione parametri"""
    from bot.backtest.optimizer import BacktestOptimizer
    
    print(f"Avvio Ottimizzazione su {args.symbol}...")
    
    optimizer = BacktestOptimizer(args.symbol)
    
    # Parametri da testare
    param_grid = {
        'sl_mult': [1.5, 2.0, 2.5],
        'tp_mult': [2.0, 3.0, 4.0],
        'score_thresh_long': [2.5, 3.0, 3.5]
    }
    
    best_params = optimizer.optimize(param_grid)
    print(f"\nOptimization completed. Best Params: {best_params}")

def cmd_train(args):
    """Avvia training ML"""
    from bot.ml.trainer import MLTrainer
    
    print(f"Avvio Training ML su {args.symbol} per {args.days} giorni...")
    trainer = MLTrainer(args.symbol)
    
    # 1. Genera dataset
    dataset = trainer.generate_dataset(days=args.days)
    if not dataset:
        print("[ERROR] Dataset vuoto o errore generazione.")
        return

    # 2. Train
    print("[ML] Avvio addestramento...")
    trainer.train(dataset)

def main():
    parser = argparse.ArgumentParser(description="Trading Bot CLI")
    subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')
    
    # Init logging
    setup_logging()
    
    # 1. TRADE command
    trade_parser = subparsers.add_parser('trade', help='Avvia bot singola istanza')
    trade_parser.add_argument('--symbol', type=str, required=True, help='Simbolo es. BTC/USDT:USDT')
    trade_parser.set_defaults(func=cmd_trade)
    
    # 2. MULTI command
    multi_parser = subparsers.add_parser('multi', help='Avvia multi-bot manager')
    multi_parser.add_argument('symbols', nargs='+', help='Lista simboli es. BTC ETH SOL')
    multi_parser.set_defaults(func=cmd_multi)
    
    # 3. BACKTEST command
    bt_parser = subparsers.add_parser('backtest', help='Esegui backtest')
    bt_parser.add_argument('--symbol', type=str, required=True, help='Simbolo')
    bt_parser.add_argument('--days', type=int, default=90, help='Giorni di backtest')
    bt_parser.set_defaults(func=cmd_backtest)
    
    # 4. OPTIMIZE command
    opt_parser = subparsers.add_parser('optimize', help='Esegui ottimizzazione')
    opt_parser.add_argument('--symbol', type=str, required=True, help='Simbolo')
    opt_parser.set_defaults(func=cmd_optimize)
    
    # 5. TRAIN command
    train_parser = subparsers.add_parser('train', help='Addestra modello ML')
    train_parser.add_argument('--symbol', type=str, required=True, help='Simbolo es. BTC')
    train_parser.add_argument('--days', type=int, default=180, help='Giorni di storico per training')
    train_parser.set_defaults(func=cmd_train)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\nOperazione interrotta dall'utente.")
    else:
        parser.print_help()

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd()) # Assicura che la root sia nel path
    main()
