from typing import Dict

class ArticleChecker:
    """Class to check the validity of articles."""

    def __init__(self):
        self.seen_ids = set()
        self.seen_urls = set()
        self.duplicate_count = 0

    def is_duplicate(self, article: Dict) -> bool:
        """Check if the article is duplicate based on ID or URL"""
        if article.id in self.seen_ids or article.url in self.seen_urls:
            self.duplicate_count += 1
            return True
        return False

    def add_article(self, article):
        """Add article ID and URL to the seen sets"""
        self.seen_ids.add(article.id)
        self.seen_urls.add(article.url)

    def get_duplicate_count(self) -> int:
        """Get the count of duplicate articles found"""
        return self.duplicate_count

    def reset(self):
        """Reset the seen sets and duplicate count"""
        self.seen_ids.clear()
        self.seen_urls.clear()
        self.duplicate_count = 0
