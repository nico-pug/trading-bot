"""
Feature Engineering per ML - Centralizzato

Estrae features tecniche per training e predizione live.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def extract_features(df: pd.DataFrame, 
                     current_idx: int = -1, 
                     score: float = 0.0, 
                     side: str = 'LONG') -> Optional[Dict[str, float]]:
    """
    Estrae features tecniche da un DataFrame al punto specificato.
    
    Args:
        df: DataFrame OHLCV con indicatori
        current_idx: Indice della candela corrente (-1 per ultima)
        score: Alpha score (input dal bot)
        side: 'LONG' o 'SHORT'
        
    Returns:
        Dizionario features o None se dati insufficienti
    """
    if df is None or len(df) < 50:
        return None
    
    # Se indice negativo, converti in positivo
    if current_idx < 0:
        current_idx = len(df) + current_idx
        
    if current_idx < 20: # Serve storico per features
        return None
        
    current = df.iloc[current_idx]
    prev_20 = df.iloc[current_idx-20:current_idx]
    
    features = {}
    
    # 1. RSI Normalizzato (0-1)
    rsi = current.get('RSI', 50)
    features['rsi'] = rsi / 100 if pd.notna(rsi) else 0.5
    
    # 2. ATR % rispetto al prezzo
    atr = current.get('ATR', 0)
    features['atr_pct'] = (atr / current['close'] * 100) if atr > 0 else 0
    
    # 3. Bollinger Band Position (0-1)
    # 0 = Lower Band, 0.5 = Mid, 1 = Upper
    bb_upper = current.get('BBU_20_2.0', current['close'])
    bb_lower = current.get('BBL_20_2.0', current['close'])
    
    if bb_upper != bb_lower:
        # Normalize to 0-1 range
        features['bb_position'] = (current['close'] - bb_lower) / (bb_upper - bb_lower)
    else:
        features['bb_position'] = 0.5
        
    # 4. VWAP Distance %
    vwap = current.get('VWAP_D', current.get('vwap', current['close']))
    features['vwap_dist'] = ((current['close'] - vwap) / vwap * 100) if vwap > 0 else 0
    
    # 5. Volume Ratio vs 20 SMA
    vol_avg = prev_20['volume'].mean()
    features['vol_ratio'] = current['volume'] / vol_avg if vol_avg > 0 else 1.0
    
    # 6. Price Momentum (ROC 5)
    momentum_period = 5
    price_lag = df['close'].iloc[current_idx - momentum_period]
    features['momentum_5'] = ((current['close'] - price_lag) / price_lag * 100)
    
    # 7. CVD Trend (se disponibile)
    if 'CVD' in df.columns:
        cvd_current = df['CVD'].iloc[current_idx]
        cvd_lag = df['CVD'].iloc[current_idx - 5]
        features['cvd_trend'] = 1 if cvd_current > cvd_lag else -1
    else:
        features['cvd_trend'] = 0
        
    # 8. Meta-features
    features['score'] = score
    features['side_long'] = 1 if side == 'LONG' else 0
    
    # Handle NaN/Inf
    for k, v in features.items():
        if pd.isna(v) or np.isinf(v):
            features[k] = 0.0
            
    return features
