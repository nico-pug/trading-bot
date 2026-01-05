"""
News Scraper (Parallelized)
RSS Feeds + NewsAPI + Fear & Greed
"""

import requests
import feedparser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from .base import BaseScraper, rate_limit

# Config (andrebbe in config.yaml)
FREE_NEWS_SOURCES = [
    "https://cointelegraph.com/rss",
    "https://cryptonews.com/news/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://news.bitcoin.com/feed"
]

class NewsScraper(BaseScraper):
    """Scraper news parallelo"""
    
    def __init__(self):
        super().__init__("news")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (TradingBot/1.0)'
        })
        
    def scrape(self, symbol: str, limit: int = 30) -> List[Dict]:
        """Scarica news da RSS in parallelo"""
        news = []
        
        # Parallel fetch RSS
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self._fetch_rss, url): url for url in FREE_NEWS_SOURCES}
            
            for future in as_completed(future_to_url):
                try:
                    items = future.result()
                    news.extend(items)
                except Exception as e:
                    pass # Ignore single feed errors
        
        # Filtra per simbolo (keyword matching semplice)
        filtered = [
            n for n in news 
            if symbol.lower() in n['title'].lower() or symbol.lower() in n['text'].lower()
        ]
        
        # Ordina per data e limita
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)
        return filtered[:limit]

    def _fetch_rss(self, url: str) -> List[Dict]:
        """Helper fetch singolo RSS"""
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:15]:
            # Data parsing robusto
            dt = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                dt = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                dt = datetime(*entry.updated_parsed[:6])
                
            items.append({
                'source': 'rss',
                'feed': url.split('/')[2],
                'title': entry.get('title', ''),
                'text': entry.get('summary', '')[:500],
                'url': entry.get('link', ''),
                'timestamp': dt
            })
        return items

class FearGreedScraper(BaseScraper):
    """Alternative.me Fear & Greed"""
    
    def __init__(self):
        super().__init__("fng")
        self.url = "https://api.alternative.me/fng/"
    
    @rate_limit(calls=10, period=60) # 10 calls/min limit
    def scrape(self, symbol: str = None, limit: int = 1) -> List[Dict]:
        try:
            resp = requests.get(self.url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()['data'][0]
                val = int(data['value'])
                # Normalize 0-100 to -1 to 1
                score = (val - 50) / 50
                
                return [{
                    'source': 'fng_index',
                    'title': f"Fear & Greed: {data['value_classification']}",
                    'text': f"Value: {val}",
                    'score': score,
                    'timestamp': datetime.now()
                }]
        except Exception as e:
            print(f"FNG Error: {e}")
        return []
