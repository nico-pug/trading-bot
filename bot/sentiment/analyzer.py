"""
Sentiment Analyzer - Aggregatore Parallelo

Orchestra multiple scrapers, esegue NLP e calcola score finale.
Features:
- Esecuzione parallela scrapers
- Deduping notizie (fuzzy matching)
- Spam filter (basic keywords)
- VADER-style sentiment analysis (semplificato)
"""

import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import sys
import os

# Fix per esecuzione come script
if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    from bot.sentiment.scrapers.news import NewsScraper, FearGreedScraper
else:
    from .scrapers.news import NewsScraper, FearGreedScraper

class SentimentAnalyzer:
    """Motore analisi sentiment parallelo"""
    
    SPAM_KEYWORDS = ['giveaway', 'airdrop', 'free', 'pump', 'join my', 'whatsapp']
    
    def __init__(self):
        self.scrapers = [
            NewsScraper(),
            FearGreedScraper()
        ]
        
        # Dizionario sentiment base (VADER style semplificato)
        self.lexicon = {
            'bull': 2.0, 'bullish': 2.0, 'moon': 1.5, 'soar': 1.5, 'surge': 1.5,
            'high': 1.0, 'gain': 1.0, 'profit': 1.0, 'up': 0.5, 'green': 0.5,
            'bear': -2.0, 'bearish': -2.0, 'crash': -2.0, 'dump': -2.0, 'collapse': -2.0,
            'low': -1.0, 'loss': -1.0, 'down': -0.5, 'red': -0.5, 'fud': -1.0
        }
        
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Ottiene sentiment aggregato per un simbolo"""
        results = []
        
        # 1. Parallel Scrape
        with ThreadPoolExecutor(max_workers=len(self.scrapers)) as executor:
            future_to_scraper = {
                executor.submit(s.scrape, symbol): s.name for s in self.scrapers
            }
            
            for future in as_completed(future_to_scraper):
                name = future_to_scraper[future]
                try:
                    items = future.result()
                    results.extend(items)
                except Exception as e:
                    print(f"Scraper error ({name}): {e}")
        
        # 2. Deduping & Filtering
        unique_items = self._deduplicate(results)
        clean_items = self._filter_spam(unique_items)
        
        # 3. Analyze Text
        scores = []
        for item in clean_items:
            # Se ha già uno score (es. FNG), usalo
            if 'score' in item:
                scores.append(item['score'])
            else:
                # Altrimenti calcola da testo
                text = f"{item['title']} {item['text']}"
                scores.append(self._calculate_text_score(text))
                
        # 4. Aggregation
        final_score = 0.0
        if scores:
            final_score = sum(scores) / len(scores)
            
        # Clipping -1 to 1
        final_score = max(-1.0, min(1.0, final_score))
        
        return {
            'symbol': symbol,
            'score': final_score,
            'volume': len(clean_items),
            'timestamp': datetime.now().isoformat(),
            'items': clean_items[:5] # Top 5 news for debug
        }
    
    def _calculate_text_score(self, text: str) -> float:
        """Calcola score sentiment semplice basato su lessico"""
        words = re.findall(r'\w+', text.lower())
        score = 0
        count = 0
        
        for word in words:
            if word in self.lexicon:
                score += self.lexicon[word]
                count += 1
                
        if count == 0:
            return 0.0
        
        # Normalize roughly
        return score / (count + 1)

    def _deduplicate(self, items: List[Dict]) -> List[Dict]:
        """Rimuove duplicati (titoli simili)"""
        seen_titles = set()
        unique = []
        
        for item in items:
            # Semplice normalizzazione titolo per deduping
            title_sim = re.sub(r'\W+', '', item['title'].lower())[:30]
            if title_sim not in seen_titles:
                seen_titles.add(title_sim)
                unique.append(item)
                
        return unique

    def _filter_spam(self, items: List[Dict]) -> List[Dict]:
        """Rimuove spam basato su keyword"""
        clean = []
        for item in items:
            text = (item['title'] + item['text']).lower()
            if not any(spam in text for spam in self.SPAM_KEYWORDS):
                clean.append(item)
        return clean

# === TEST ===
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    print("Analisi sentiment BTC in corso (parallelo)...")
    res = analyzer.analyze_symbol("BTC")
    
    print(f"\nRisultato {res['symbol']}:")
    print(f"Score: {res['score']:.2f}")
    print(f"Volume: {res['volume']}")
    print("Top News:")
    for n in res['items']:
        print(f"- {n['title']} ({n.get('source', 'unknown')})")
