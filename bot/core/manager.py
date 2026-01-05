"""
Process Manager - Multi-Bot Supervisor
Gestisce l'avvio e il monitoraggio di più processi bot.
"""

import subprocess
import sys
import time
import os
import signal
from typing import Dict, List
from bot.utils.logger import get_logger

class ProcessManager:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logger = get_logger("ProcessManager")
        self.running = True
        
        signal.signal(signal.SIGINT, self.stop_all)
        signal.signal(signal.SIGTERM, self.stop_all)

    def start_bot(self, symbol: str):
        """Avvia un bot processo figlio"""
        if symbol in self.processes and self.processes[symbol].poll() is None:
            self.logger.warning(f"Bot {symbol} già in esecuzione.")
            return

        cmd = [sys.executable, "-m", "bot.main", "trade", "--symbol", symbol]
        
        try:
            # Avvia processo (eredita stdout/stderr così vediamo i log)
            proc = subprocess.Popen(
                cmd,
                stdout=None, 
                stderr=None,
                cwd=os.getcwd()
            )
            self.processes[symbol] = proc
            self.logger.info(f"Avviato bot {symbol} (PID: {proc.pid})")
        except Exception as e:
            self.logger.error(f"Errore avvio bot {symbol}: {e}")

    def monitor(self):
        """Loop di monitoraggio e restart automatico"""
        self.logger.info("Monitoraggio processi avviato...")
        while self.running:
            for symbol, proc in list(self.processes.items()):
                ret_code = proc.poll()
                if ret_code is not None:
                    # Il processo è terminato
                    self.logger.warning(f"Bot {symbol} terminato (Exit Code: {ret_code}). Riavvio tra 5s...")
                    
                    del self.processes[symbol]
                    time.sleep(5)
                    self.start_bot(symbol)
                    
            time.sleep(10)

    def stop_all(self, signum=None, frame=None):
        """Ferma tutti i processi"""
        self.logger.info("Arresto di tutti i bot...")
        self.running = False
        for symbol, proc in self.processes.items():
            try:
                proc.terminate()
                self.logger.info(f"Terminato {symbol}")
            except:
                pass
        sys.exit(0)

    def launch_multi(self, symbols: List[str]):
        """Lancia lista di simboli"""
        for s in symbols:
            self.start_bot(s)
        
        try:
            self.monitor()
        except KeyboardInterrupt:
            self.stop_all()
