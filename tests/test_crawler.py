
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from bs4 import BeautifulSoup
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
subparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, subparent_dir)

# Import the crawler to test
from news_data_crawling.crawlers.news_crawler import NewsCrawler
from news_data_crawling.crawlers.news_models import CrawledData

class TestNewsCrawler:
    """Test NewsCrawler functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.crawler = NewsCrawler(es_manager=None)
        
        # Sample RSS item
        self.sample_rss_item = """
        <item>
            <title>Test Article Title</title>
            <link>https://example.com/test-article</link>
            <description>Test article description</description>
            <pubDate>Mon, 01 Jan 2023 12:00:00 GMT</pubDate>
            <category>Technology</category>
        </item>
        """

    def test_get_country_info_known_source(self):
        """Test country info for known sources"""
        country, iso_code = self.crawler.get_country_info("Kompas", "Indonesia")
        assert country == "Indonesia"
        assert iso_code == "ID"
        
        country, iso_code = self.crawler.get_country_info("BBC", "UK")
        assert country == "United Kingdom"
        assert iso_code == "GB"

    def test_get_country_info_unknown_source(self):
        """Test country info for unknown sources"""
        country, iso_code = self.crawler.get_country_info("UnknownSource", "Indonesia")
        assert country == "Indonesia"
        assert iso_code == "ID"
        
        country, iso_code = self.crawler.get_country_info("UnknownSource", "Global")
        assert country == "International"
        assert iso_code == "ZZ"

    def test_clean_html_content(self):
        """Test HTML content cleaning"""
        html_content = """
        <html>
            <script>alert('test');</script>
            <style>body { color: red; }</style>
            <div>Hello <b>World</b>!</div>
        </html>
        """
        
        result = self.crawler.clean_html_content(html_content)
        assert "Hello World!" in result
        assert "alert" not in result
        assert "color: red" not in result

    def test_clean_html_content_empty(self):
        """Test HTML content cleaning with empty input"""
        result = self.crawler.clean_html_content("")
        assert result == ""
        
        result = self.crawler.clean_html_content(None)
        assert result == ""

    @patch('news_data_crawling.crawlers.news_crawler.requests.get')
    def test_extract_full_content_success(self, mock_get):
        """Test successful content extraction"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = """
        <html>
            <article>
                <p>This is the article content.</p>
                <p>It has multiple paragraphs.</p>
            </article>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = self.crawler.extract_full_content("https://example.com/article")
        assert "article content" in result
        assert "multiple paragraphs" in result

    @patch('news_data_crawling.crawlers.news_crawler.requests.get')
    def test_extract_full_content_failure(self, mock_get):
        """Test content extraction failure"""
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        result = self.crawler.extract_full_content("https://example.com/article")
        assert result == ""

    @patch('news_data_crawling.crawlers.news_crawler.requests.get')
    def test_extract_full_content_non_200(self, mock_get):
        """Test content extraction with non-200 response"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.crawler.extract_full_content("https://example.com/article")
        assert result == ""

    def test_parse_rss_item_valid(self):
        """Test parsing valid RSS item"""
        soup = BeautifulSoup(self.sample_rss_item, "xml")
        item = soup.find("item")
        
        with patch.object(self.crawler, 'extract_full_content', return_value="Full content"):
            with patch.object(self.crawler.article_checker, 'is_duplicate', return_value=False):
                result = self.crawler._parse_rss_item(item, "TestSource", "Indonesia")
                
                assert result is not None
                assert result.title == "Test Article Title"
                assert result.url == "https://example.com/test-article"
                assert result.source_name == "TestSource"
                assert result.metadata["region"] == "Indonesia"

    def test_parse_rss_item_missing_fields(self):
        """Test parsing RSS item with missing fields"""
        incomplete_rss = """
        <item>
            <title>Test Title</title>
            <!-- Missing link and description -->
        </item>
        """
        
        soup = BeautifulSoup(incomplete_rss, "xml")
        item = soup.find("item")
        
        result = self.crawler._parse_rss_item(item, "TestSource", "Indonesia")
        assert result is not None
        assert result.title == "Test Title"
        assert result.url == ""

    def test_parse_rss_item_exception(self):
        """Test parsing RSS item that causes exception"""
        # Create a malformed item that will cause an exception
        class MalformedItem:
            def find(self, *args, **kwargs):
                raise Exception("Test exception")
        
        result = self.crawler._parse_rss_item(MalformedItem(), "TestSource", "Indonesia")
        assert result is None

    def test_crawl_rss_success(self):
        """Test successful RSS crawling"""
        mock_response = Mock()
        mock_response.content = """
        <rss>
            <channel>
                <item>
                    <title>Article 1</title>
                    <link>https://example.com/1</link>
                    <description>Desc 1</description>
                </item>
                <item>
                    <title>Article 2</title>
                    <link>https://example.com/2</link>
                    <description>Desc 2</description>
                </item>
            </channel>
        </rss>
        """

        with patch.object(self.crawler, 'get_page', return_value=mock_response):
            with patch.object(self.crawler, '_parse_rss_item', return_value=Mock()):
                with patch.object(self.crawler.article_checker, 'is_duplicate', return_value=False):
                    results = self.crawler.crawl_rss("TestSource", "https://example.com/rss", "Indonesia")
                    assert len(results) == 2

    @patch('news_data_crawling.crawlers.news_crawler.requests.get')
    def test_crawl_rss_failure(self, mock_get):
        """Test RSS crawling failure"""
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        results = self.crawler.crawl_rss("TestSource", "https://example.com/rss", "Indonesia")
        assert results == []

    def test_remove_unwanted_elements(self):
        """Test removal of unwanted HTML elements"""
        html_content = """
        <div>
            <script>alert('test');</script>
            <style>body { color: red; }</style>
            <nav>Navigation</nav>
            <header>Header</header>
            <footer>Footer</footer>
            <aside>Sidebar</aside>
            <advertisement>Ad</advertisement>
            <article>Real content here</article>
        </div>
        """
        
        soup = BeautifulSoup(html_content, "html.parser")
        self.crawler._remove_unwanted_elements(soup)
        
        result_text = soup.get_text()
        assert "Real content here" in result_text
        assert "alert" not in result_text
        assert "Navigation" not in result_text
        assert "Header" not in result_text

    def test_extract_content_from_soup_with_article(self):
        """Test content extraction from soup with article tag"""
        html_content = """
        <div>
            <article>
                <p>First paragraph of the article.</p>
                <p>Second paragraph with more content.</p>
            </article>
        </div>
        """
        
        soup = BeautifulSoup(html_content, "html.parser")
        result = self.crawler._extract_content_from_soup(soup)
        
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_extract_content_from_soup_with_paragraphs(self):
        """Test content extraction from soup with paragraphs"""
        html_content = """
        <div class="content">
            <p>This is the first paragraph of the content.</p>
            <p>This is the second paragraph that should be included.</p>
            <div>Advertisement: Buy now!</div>
        </div>
        """
        
        soup = BeautifulSoup(html_content, "html.parser")
        result = self.crawler._extract_content_from_soup(soup)
        
        assert "first paragraph" in result
        assert "second paragraph" in result
        assert "Advertisement" not in result

    def test_generate_id_consistency(self):
        """Test that ID generation is consistent for same inputs"""
        url1 = "https://example.com/article"
        title1 = "Test Article"
        
        id1 = self.crawler.generate_id(url1, title1)
        id2 = self.crawler.generate_id(url1, title1)
        
        assert id1 == id2
        
        # Different inputs should generate different IDs
        id3 = self.crawler.generate_id("https://example.com/different", title1)
        assert id1 != id3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])