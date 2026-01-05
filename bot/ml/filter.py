"""
ML Filter - Filtro segnali e Drift Detection

Applica il modello ML ai segnali e monitora le performance nel tempo.
Dataset drift detection implementata.
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import deque
from bot.ml.features import extract_features

class MLFilter:
    """Applicazione modello ML e monitoring"""
    
    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        self.model = None
        self.features = []
        self.enabled = False
        self.threshold = 0.55 # Default conservativo
        
        # Drift tracking
        self.prediction_history = deque(maxlen=50) # Ultime 50 predizioni
        self.recent_accuracy = deque(maxlen=20)   # Ultimi 20 outcome reali
        
        self._load_model()
        
    def _load_model(self):
        path = f"ml_models/ml_model_{self.symbol}.pkl"
        if not os.path.exists(path):
            # Fallback a modello generico o BTC se specifico non esiste
            path = "ml_models/ml_model_BTC.pkl"
            
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.features = data['features']
                    self.enabled = True
                    print(f"[ML] Modello caricato: {path}")
            except Exception as e:
                print(f"[ML] Errore load modello: {e}")
        else:
            print("[ML] Nessun modello trovato. ML Filter disabilitato.")

    def check_signal(self, df, idx, score, side) -> float:
        """
        Valuta un segnale.
        Returns: probabilità (0-1)
        """
        if not self.enabled:
            return 1.0 # Pass-through se disabilitato
            
        # Estrai features
        feats_dict = extract_features(df, idx, score, side)
        if not feats_dict:
            return 0.5
            
        # Ordina come nel training
        try:
            vector = [feats_dict.get(f, 0) for f in self.features]
            prob = self.model.predict([vector])[0]
            
            # Track drift
            self.prediction_history.append(prob)
            
            return prob
        except Exception as e:
            print(f"[ML] Predict error: {e}")
            return 0.5

    def record_outcome(self, prob: float, is_win: bool):
        """Registra risultato reale per drift detection"""
        prediction_correct = (prob > 0.5) == is_win
        self.recent_accuracy.append(1 if prediction_correct else 0)
        
        self._check_drift()

    def _check_drift(self):
        """Controlla se accuratezza recente è degradata"""
        if len(self.recent_accuracy) >= 20:
            avg_acc = np.mean(self.recent_accuracy)
            if avg_acc < 0.40: # Sotto 40% è preoccupante
                print(f"[ML] DRIFT DETECTED! Accuracy drop to {avg_acc:.1%}")
                # Potenziale azione: alzare threshold o disabilitare filtro
                self.threshold = min(0.8, self.threshold + 0.05)
