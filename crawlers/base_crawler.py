import time
import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

class BaseCrawler:
    """Base class for crawlers with session handling, retries, and ID generation"""

    def __init__(self, name: str, delay: float = 1.0):
        self.name = name
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                            'AppleWebKit/537.36 (KHTML, like Gecko) '
                            'Chrome/58.0.3029.110 Safari/537.3'
        })
        retries = Retry(
            total=10,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def get_page(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Fetch a web page with error handling and rate limiting"""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def generate_id(self, url: str, title: str) -> str:
        """Generate a unique identifier for a URL + title"""
        content_str = f"{url}_{title}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return hashlib.md5(content_str.encode()).hexdigest()
