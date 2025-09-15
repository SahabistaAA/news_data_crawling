import re
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Optional

from news_models import CrawledData
from base_crawler import BaseCrawler
from article_checker import ArticleChecker

logger = logging.getLogger(__name__)


class NewsCrawler(BaseCrawler):
    """Crawler for Indonesian and Global news websites."""

    def __init__(self, es_manager=None):
        super().__init__("NewsCrawler", delay=2.0)
        self.es_manager = es_manager
        self.article_checker = ArticleChecker()

        # Source-to-country mapping (name → (country, ISO code))
        self.source_country_map = {
            # Indonesian sources
            'Kompas': ("Indonesia", "ID"),
            'Detik': ("Indonesia", "ID"),
            'Antara': ("Indonesia", "ID"),
            'Tempo': ("Indonesia", "ID"),
            'CNN Indonesia': ("Indonesia", "ID"),
            'Liputan6': ("Indonesia", "ID"),
            'Okezone': ("Indonesia", "ID"),
            'Republika': ("Indonesia", "ID"),
            'Suara': ("Indonesia", "ID"),
            'Tribunnews': ("Indonesia", "ID"),
            'Kontan': ("Indonesia", "ID"),
            'CNBC Indonesia': ("Indonesia", "ID"),
            'JawaPos': ("Indonesia", "ID"),
            'Kumparan': ("Indonesia", "ID"),
            'Tirto': ("Indonesia", "ID"),

            # Global sources
            'BBC': ("United Kingdom", "GB"),
            'CNN': ("United States", "US"),
            'Reuters': ("United Kingdom", "GB"),
            'Guardian': ("United Kingdom", "GB"),
            'Associated Press': ("United States", "US"),
            'Al Jazeera': ("Qatar", "QA"),
            'Deutsche Welle': ("Germany", "DE"),
            'France24': ("France", "FR"),
            'RT': ("Russia", "RU"),
            'Sky News': ("United Kingdom", "GB"),
            'NPR': ("United States", "US"),
            'ABC News': ("Australia", "AU"),
            'Washington Post': ("United States", "US"),
        }

    # ==========
    # Helpers
    # ==========

    def get_country_info(self, source_name: str, region: str) -> tuple:
        """Resolve country name and ISO code based on source or region."""
        if source_name in self.source_country_map:
            return self.source_country_map[source_name]

        if region.lower() == "indonesia":
            return ("Indonesia", "ID")
        else:
            return ("International", "ZZ")  # ZZ = unspecified/unknown country

    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and extract plain text only."""
        if not html_content:
            return ""

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logger.warning(f"Error cleaning HTML content: {e}")
            return BeautifulSoup(html_content, "html.parser").get_text(separator=' ', strip=True)

    def extract_full_content(self, url: str) -> str:
        """Extract full article content from the article URL."""
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if not response or response.status_code != 200:
                return ""

            soup = BeautifulSoup(response.content, "html.parser")
            self._remove_unwanted_elements(soup)
            text = self._extract_content_from_soup(soup)
            return text

        except Exception as e:
            logger.warning(f"Error extracting full content from {url}: {e}")
            return ""

    def _remove_unwanted_elements(self, soup):
        """Remove unwanted elements from the soup."""
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement", "ads"]):
            element.decompose()

    def _extract_content_from_soup(self, soup) -> str:
        """Try to extract main content from soup using various strategies."""
        content_selectors = [
            'article',
            '.article-content',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-body',
            '.story-body',
            '[data-testid="article-body"]',
            '.article__body',
            '.news-content',
            '.detail-content',
            '.read-content',
            'div[itemprop="articleBody"]',
            '.artikel-content',
            '.news-article-content'
        ]

        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                text = content_element.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    return re.sub(r'\s+', ' ', text).strip()

        # Fallback: try to find paragraphs
        main_content = soup.find('main') or soup.find('div', class_=re.compile(r'(content|article|post)'))
        if main_content:
            paragraphs = main_content.find_all('p')
            if paragraphs:
                text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                if len(text) > 100:
                    return re.sub(r'\s+', ' ', text).strip()

        # Last fallback: extract all text
        all_text = soup.get_text(separator=' ', strip=True)
        all_text = re.sub(r'\s+', ' ', all_text).strip()
        return all_text + ('...' if len(all_text) > 2000 else '')

    # ==========
    # Crawling
    # ==========

    def crawl_rss(self, source_name: str, feed_url: str, region: str) -> List[CrawledData]:
        """Crawl RSS feed and extract articles."""
        articles = []
        try:
            response = self.get_page(feed_url)
            if not response:
                return articles

            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")

            for item in items:
                article = self._parse_rss_item(item, source_name, region)
                if article and not self.article_checker.is_duplicate(article):
                    self.article_checker.add_article(article)
                    articles.append(article)
        except Exception as e:
            logger.error(f"Error crawling feed {feed_url}: {e}")

        logger.info(f"Crawled {len(articles)} articles from {source_name} ({region})")
        return articles

    def _parse_rss_item(self, item, source_name: str, region: str) -> Optional[CrawledData]:
        """Parse a single RSS item and return a CrawledData object or None."""
        try:
            title = item.find("title")
            link = item.find("link")
            description = item.find("description")
            pub_date = item.find("pubDate") or item.find("date")
            category = item.find("category")

            full_content = ""
            if link and link.text.strip():
                full_content = self.extract_full_content(link.text.strip())

            country, iso_code = self.get_country_info(source_name, region)

            article = CrawledData(
                id=self.generate_id(
                    link.text.strip() if link else "",
                    title.text.strip() if title else ""
                ),
                source_type="news",
                source_name=source_name,
                title=title.text.strip() if title else "No title",
                content=self.clean_html_content(
                    description.text.strip() if description else ""
                ),
                url=link.text.strip() if link else "",
                timestamp=datetime.now().isoformat(),
                metadata={
                    "pub_date": pub_date.text if pub_date else datetime.now().isoformat(),
                    "category": category.text if category else "general",
                    "region": region,
                    "country": country,
                    "country_code": iso_code,
                    "language": "id" if region.lower() == "indonesia" else "en",
                },
                full_content=full_content
            )
            return article
        except Exception as e:
            logger.error(f"Error processing article from {source_name}: {e}")
            return None
