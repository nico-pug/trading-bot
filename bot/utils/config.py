"""
Configuration Loader for Trading Bot
Carica e gestisce la configurazione da config.yaml

Usage:
    from bot.utils.config import load_config, Config
    
    config = load_config()
    capital = config.trading.capital
    sl_mult = config.get_symbol_config('BTC').sl_multiplier
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


# Path al file di configurazione
# __file__ = bot/utils/config.py -> parent.parent.parent = trading_bot/
CONFIG_DIR = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.yaml"


@dataclass
class TradingConfig:
    """Configurazione trading"""
    capital: float = 10000.0
    risk_per_trade: float = 0.01
    max_drawdown_limit: float = 0.20
    leverage_cap: int = 5
    score_threshold_long: float = 3.0
    score_threshold_short: float = -3.0
    sl_multiplier: float = 2.0
    tp_multiplier: float = 3.0
    fee_rate: float = 0.0004
    slippage: float = 0.0005


@dataclass
class SymbolConfig:
    """Configurazione per singolo simbolo"""
    sl_multiplier: float = 2.0
    tp_multiplier: float = 3.0
    score_threshold_long: float = 3.0
    score_threshold_short: float = -3.0


@dataclass
class APIConfig:
    """Configurazione API"""
    enable_rate_limit: bool = True
    rate_limit_requests: int = 1200
    timeout: int = 30
    retries: int = 3


@dataclass
class MLConfig:
    """Configurazione Machine Learning"""
    enabled: bool = True
    model_dir: str = "ml_models"
    min_probability: float = 0.40
    threshold: float = 0.5
    features: List[str] = field(default_factory=lambda: [
        'rsi', 'atr_pct', 'bb_position', 'vwap_distance',
        'cvd_trend', 'volume_ratio', 'price_momentum',
        'funding_rate', 'ls_ratio', 'score'
    ])


@dataclass
class SentimentConfig:
    """Configurazione Sentiment"""
    enabled: bool = True
    cache_duration_seconds: int = 1800
    weights: Dict[str, float] = field(default_factory=lambda: {
        'reddit': 0.4, 'news': 0.4, 'fear_greed': 0.2
    })
    bullish_threshold: float = 0.15
    bearish_threshold: float = -0.15


@dataclass
class BacktestConfig:
    """Configurazione Backtest"""
    days: int = 180
    timeframe: str = "15m"
    initial_capital: float = 10000.0
    walk_forward_enabled: bool = True
    train_ratio: float = 0.7
    n_folds: int = 3


@dataclass
class SystemConfig:
    """Configurazione Sistema"""
    scan_interval: int = 60


@dataclass
class LoggingConfig:
    """Configurazione Logging"""
    level: str = "INFO"
    file: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    console: bool = True
    use_colors: bool = True


class Config:
    """
    Classe principale di configurazione.
    Carica e fornisce accesso tipizzato a tutti i parametri.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._raw = config_dict or {}
        
        # Parse sezioni
        self.trading = self._parse_trading()
        self.api = self._parse_api()
        self.ml = self._parse_ml()
        self.sentiment = self._parse_sentiment()
        self.backtest = self._parse_backtest()
        self.system = self._parse_system()
        self.logging = self._parse_logging()
        
        # Configurazioni per simbolo
        self._symbol_configs: Dict[str, SymbolConfig] = {}
        self._parse_symbols()
    
    def _parse_trading(self) -> TradingConfig:
        """Parse sezione trading"""
        t = self._raw.get('trading', {})
        return TradingConfig(
            capital=t.get('capital', 10000.0),
            risk_per_trade=t.get('risk_per_trade', 0.01),
            max_drawdown_limit=t.get('max_drawdown_limit', 0.20),
            leverage_cap=t.get('leverage_cap', 5),
            score_threshold_long=t.get('score_threshold_long', 3.0),
            score_threshold_short=t.get('score_threshold_short', -3.0),
            sl_multiplier=t.get('sl_multiplier', 2.0),
            tp_multiplier=t.get('tp_multiplier', 3.0),
            fee_rate=t.get('fee_rate', 0.0004),
            slippage=t.get('slippage', 0.0005)
        )
    
    def _parse_api(self) -> APIConfig:
        """Parse sezione API"""
        a = self._raw.get('api', {}).get('binance', {})
        return APIConfig(
            enable_rate_limit=a.get('enable_rate_limit', True),
            rate_limit_requests=a.get('rate_limit_requests', 1200),
            timeout=a.get('timeout', 30),
            retries=a.get('retries', 3)
        )
    
    def _parse_ml(self) -> MLConfig:
        """Parse sezione ML"""
        m = self._raw.get('ml', {})
        default_features = [
            'rsi', 'atr_pct', 'bb_position', 'vwap_distance',
            'cvd_trend', 'volume_ratio', 'price_momentum',
            'funding_rate', 'ls_ratio', 'score'
        ]
        return MLConfig(
            enabled=m.get('enabled', True),
            model_dir=m.get('model_dir', 'ml_models'),
            min_probability=m.get('min_probability', 0.40),
            threshold=m.get('threshold', 0.5),
            features=m.get('features', default_features)
        )
    
    def _parse_sentiment(self) -> SentimentConfig:
        """Parse sezione sentiment"""
        s = self._raw.get('sentiment', {})
        thresholds = s.get('thresholds', {})
        default_weights = {'reddit': 0.4, 'news': 0.4, 'fear_greed': 0.2}
        return SentimentConfig(
            enabled=s.get('enabled', True),
            cache_duration_seconds=s.get('cache_duration_seconds', 1800),
            weights=s.get('weights', default_weights),
            bullish_threshold=thresholds.get('bullish', 0.15),
            bearish_threshold=thresholds.get('bearish', -0.15)
        )
    
    def _parse_backtest(self) -> BacktestConfig:
        """Parse sezione backtest"""
        b = self._raw.get('backtest', {})
        wf = b.get('walk_forward', {})
        return BacktestConfig(
            days=b.get('days', 180),
            timeframe=b.get('timeframe', '15m'),
            initial_capital=b.get('initial_capital', 10000.0),
            walk_forward_enabled=wf.get('enabled', True),
            train_ratio=wf.get('train_ratio', 0.7),
            n_folds=wf.get('n_folds', 3)
        )
    
    def _parse_system(self) -> SystemConfig:
        """Parse sezione system"""
        s = self._raw.get('system', {})
        return SystemConfig(
            scan_interval=s.get('scan_interval', 60)
        )
    
    def _parse_logging(self) -> LoggingConfig:
        """Parse sezione logging"""
        l = self._raw.get('logging', {})
        return LoggingConfig(
            level=l.get('level', 'INFO'),
            file=l.get('file'),
            max_bytes=l.get('max_bytes', 10485760),
            backup_count=l.get('backup_count', 5),
            console=l.get('console', True),
            use_colors=l.get('use_colors', True)
        )
    
    def _parse_symbols(self) -> None:
        """Parse configurazioni per simbolo"""
        symbols = self._raw.get('symbols', {})
        for symbol, cfg in symbols.items():
            self._symbol_configs[symbol.upper()] = SymbolConfig(
                sl_multiplier=cfg.get('sl_multiplier', self.trading.sl_multiplier),
                tp_multiplier=cfg.get('tp_multiplier', self.trading.tp_multiplier),
                score_threshold_long=cfg.get('score_threshold_long', self.trading.score_threshold_long),
                score_threshold_short=cfg.get('score_threshold_short', self.trading.score_threshold_short)
            )
    
    def get_symbol_config(self, symbol: str) -> SymbolConfig:
        """
        Ottiene configurazione per un simbolo specifico.
        Se non esiste, usa i default.
        """
        # Estrai base symbol (es. BTCUSDT -> BTC)
        base = symbol.upper().replace('USDT', '').replace('/USDT:USDT', '')
        
        if base in self._symbol_configs:
            return self._symbol_configs[base]
        
        # Default
        return SymbolConfig(
            sl_multiplier=self.trading.sl_multiplier,
            tp_multiplier=self.trading.tp_multiplier,
            score_threshold_long=self.trading.score_threshold_long,
            score_threshold_short=self.trading.score_threshold_short
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Accesso raw alla configurazione"""
        keys = key.split('.')
        value = self._raw
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value


# Singleton per configurazione globale
_config_instance: Optional[Config] = None


def load_config(config_file: str = None, force_reload: bool = False) -> Config:
    """
    Carica la configurazione da file YAML.
    
    Args:
        config_file: Path al file di configurazione (default: config.yaml nella root)
        force_reload: Se True, ricarica anche se già caricata
        
    Returns:
        Oggetto Config con tutti i parametri
    """
    global _config_instance
    
    if _config_instance is not None and not force_reload:
        return _config_instance
    
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    
    config_path = Path(config_file)
    
    if not config_path.exists():
        print(f"[WARNING] Config file not found: {config_path}")
        print("[INFO] Using default configuration")
        _config_instance = Config({})
        return _config_instance
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        _config_instance = Config(config_dict)
        print(f"[OK] Configuration loaded from {config_path}")
        return _config_instance
        
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        print("[INFO] Using default configuration")
        _config_instance = Config({})
        return _config_instance


def get_config() -> Config:
    """Ottiene la configurazione corrente (lazy load)"""
    global _config_instance
    if _config_instance is None:
        return load_config()
    return _config_instance


# === Test del modulo ===
if __name__ == "__main__":
    config = load_config()
    
    print("\n=== CONFIG TEST ===")
    print(f"Capital: ${config.trading.capital:,.2f}")
    print(f"Risk per trade: {config.trading.risk_per_trade:.1%}")
    print(f"Fee rate: {config.trading.fee_rate:.4%}")
    print(f"ML enabled: {config.ml.enabled}")
    print(f"Log level: {config.logging.level}")
    
    btc_cfg = config.get_symbol_config('BTCUSDT')
    print(f"\nBTC Config:")
    print(f"  SL mult: {btc_cfg.sl_multiplier}")
    print(f"  TP mult: {btc_cfg.tp_multiplier}")
    
    print("\n✅ Config test completato!")
