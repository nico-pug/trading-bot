"""
ML Trainer - LightGBM Training pipeline (Time-Series Aware)

Features:
- Time-Series Split (no shuffle, no leakage)
- Feature Engineering centralizzato
- Dataset balancing (scale_pos_weight)
- Early stopping
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# Fix path
sys.path.insert(0, os.getcwd())

# Check LightGBM
try:
    import lightgbm as lgb
except ImportError:
    print("[ERROR] LightGBM non installato.")
    sys.exit(1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bot.backtest.engine import BacktestEngine
from bot.ml.features import extract_features
# from bot.core.alpha_engine import AlphaEngine  # Import dinamico per evitare cicli

class MLTrainer:
    """Trainer per modello di filtraggio trade"""
    
    MODEL_DIR = "ml_models"
    
    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        
    def generate_dataset(self, days: int = 180) -> List[Dict]:
        """Genera dataset da backtest storico"""
        print(f"[ML] Generazione dataset {self.symbol}, {days} giorni...")
        
        # Init components
        # Import locale per evitare circular deps
        try:
            from bot.core.alpha_engine import AlphaEngine
        except ImportError:
            return []
            
        alpha = AlphaEngine()
        
        # Download dati backtest
        engine = BacktestEngine(symbol=f"{self.symbol}/USDT:USDT")
        try:
            df = engine.download_historical_data(days=days, use_cache=True)
        except Exception as e:
            print(f"[ML] Errore download dati: {e}")
            return []
            
        dataset = []
        
        # Parametri mock
        # In futuro dovrebbero venire da config ottimizzata
        sl_mult = 2.0
        tp_mult = 3.0
        thresh = 1.5 # Abbassato per catturare più segnali con MockFeed
        
        total_candles = len(df)
        
        # Scansione
        for i in range(50, total_candles - 20):
            current_slice = df.iloc[:i+1]
            current = current_slice.iloc[-1]
            
            # 1. Calcola Alpha Score
            # Mock deriv/dom per velocità (in backtest reale si userebbero dati veri se disponibili)
            deriv = {'price': current['close'], 'funding_rate': 0.0, 'open_interest': 0, 'basis_pct': 0}
            dom = {'imbalance': 0}
            ls_ratio = {'long_short_ratio': 1.0}
            
            class MockFeed:
                def identify_order_blocks(self, df): return []
                def identify_fvg(self, df): return []
                def detect_liquidity_grab(self, df): return None
                def detect_mss(self, df): return None
            
            score, _ = alpha.analyze(current_slice, deriv, dom, ls_ratio, MockFeed())
            
            # 2. Check Signal
            if abs(score) >= thresh:
                side = 'LONG' if score > 0 else 'SHORT'
                
                # 3. Simula Outcome
                outcome = self._simulate_trade_outcome(df, i, side, sl_mult, tp_mult)
                
                if outcome is not None:
                    # 4. Estrai Features
                    feats = extract_features(df, i, score, side)
                    if feats:
                        feats['target'] = outcome
                        dataset.append(feats)
                        
        print(f"[ML] Generati {len(dataset)} campioni.")
        
        # Salvataggio dataset per analisi future
        if dataset:
            os.makedirs("data/datasets", exist_ok=True)
            pd.DataFrame(dataset).to_csv(f"data/datasets/ml_dataset_{self.symbol}.csv", index=False)
            print(f"[ML] Dataset salvato in data/datasets/ml_dataset_{self.symbol}.csv")
            
        return dataset

    def _simulate_trade_outcome(self, df, idx, side, sl_mult, tp_mult) -> Optional[int]:
        """Simula trade (1=Win, 0=Loss)"""
        current = df.iloc[idx]
        atr = current.get('ATR', 0)
        if atr <= 0: return None
        
        entry = current['close']
        
        if side == 'LONG':
            sl = entry - (atr * sl_mult)
            tp = entry + (atr * tp_mult)
        else:
            sl = entry + (atr * sl_mult)
            tp = entry - (atr * tp_mult)
            
        # Look forward
        for j in range(idx+1, min(idx+50, len(df))):
            future = df.iloc[j]
            
            if side == 'LONG':
                if future['low'] <= sl: return 0
                if future['high'] >= tp: return 1
            else:
                if future['high'] >= sl: return 0
                if future['low'] <= tp: return 1
                
        return None # Time limit exceeded

    def train(self, dataset: List[Dict]):
        """Traina LightGBM con split temporale"""
        if not dataset:
            print("[ML] Dataset vuoto.")
            return
            
        df = pd.DataFrame(dataset)
        X = df.drop(columns=['target'])
        y = df['target']
        feature_names = X.columns.tolist()
        
        # Temporal Split (primo 80% train, ultimo 20% test)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"[ML] Split temporale: {len(X_train)} train, {len(X_test)} test")
        print(f"[ML] Win rate train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")
        
        # LightGBM Data
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        
        # Class Imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        scale_pos_weight = min(scale_pos_weight, 2.0) # Cap weight
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 20,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1
        }
        
        print("[ML] Training...")
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # Eval
        preds = model.predict(X_test)
        preds_class = (preds > 0.5).astype(int)
        
        acc = accuracy_score(y_test, preds_class)
        prec = precision_score(y_test, preds_class, zero_division=0)
        
        print(f"\n[ML] Risultati Test Set:")
        print(f"     Accuracy : {acc:.1%}")
        print(f"     Precision: {prec:.1%}")
        
        # Save
        self._save_model(model, feature_names)

    def _save_model(self, model, features):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        path = os.path.join(self.MODEL_DIR, f"ml_model_{self.symbol}.pkl")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': features,
                'timestamp': datetime.now()
            }, f)
        print(f"[ML] Modello salvato: {path}")

if __name__ == "__main__":
    # Setup per esecuzione script
    sys.path.insert(0, os.getcwd())
    
    trainer = MLTrainer("BTC")
    data = trainer.generate_dataset(days=120)
    trainer.train(data)
