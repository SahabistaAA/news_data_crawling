import logging
import re
from typing import Dict, List, Any, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


class TopicClassifier:
    """Topic classification for news articles"""

    def __init__(self):
        self.topic_keywords = self._load_topic_keywords()
        self.stop_words = set()

        try:
            # Download NLTK data if not available
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK data not found. Installing...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def _load_topic_keywords(self) -> Dict[str, List[str]]:
        """Load predefined topic keywords"""
        return {
            'Technology': [
                'technology', 'tech', 'software', 'hardware', 'computer', 'digital',
                'internet', 'web', 'mobile', 'app', 'application', 'programming',
                'coding', 'developer', 'ai', 'artificial intelligence', 'machine learning',
                'blockchain', 'cryptocurrency', 'cybersecurity', 'data', 'analytics',
                'cloud', 'iot', 'robotics', 'automation', '5g', 'quantum'
            ],
            'Business': [
                'business', 'company', 'corporate', 'market', 'finance', 'financial',
                'investment', 'stock', 'trading', 'economy', 'economic', 'commerce',
                'industry', 'enterprise', 'startup', 'venture', 'merger', 'acquisition',
                'revenue', 'profit', 'loss', 'quarterly', 'annual', 'growth'
            ],
            'Politics': [
                'politics', 'political', 'government', 'election', 'president', 'minister',
                'parliament', 'congress', 'senate', 'policy', 'law', 'legislation',
                'diplomatic', 'international', 'foreign', 'domestic', 'campaign',
                'democracy', 'republic', 'conservative', 'liberal', 'party'
            ],
            'Health': [
                'health', 'medical', 'medicine', 'hospital', 'doctor', 'patient',
                'disease', 'treatment', 'vaccine', 'pandemic', 'epidemic', 'virus',
                'infection', 'healthcare', 'pharmaceutical', 'drug', 'clinical',
                'therapy', 'diagnosis', 'symptom', 'cure', 'wellness'
            ],
            'Sports': [
                'sports', 'sport', 'game', 'match', 'tournament', 'championship',
                'athlete', 'player', 'team', 'league', 'football', 'basketball',
                'baseball', 'soccer', 'tennis', 'golf', 'olympic', 'medal', 'coach',
                'stadium', 'arena', 'score', 'victory', 'defeat'
            ],
            'Entertainment': [
                'entertainment', 'movie', 'film', 'cinema', 'actor', 'actress',
                'celebrity', 'music', 'song', 'album', 'concert', 'award', 'hollywood',
                'broadway', 'theater', 'tv', 'television', 'show', 'series', 'episode',
                'streaming', 'netflix', 'amazon', 'disney'
            ],
            'Science': [
                'science', 'scientific', 'research', 'study', 'experiment', 'discovery',
                'physics', 'chemistry', 'biology', 'astronomy', 'space', 'nasa',
                'universe', 'planet', 'climate', 'environment', 'ecology', 'evolution',
                'genetics', 'neuroscience', 'psychology', 'anthropology'
            ],
            'Education': [
                'education', 'school', 'university', 'college', 'student', 'teacher',
                'academic', 'learning', 'teaching', 'curriculum', 'degree', 'graduate',
                'undergraduate', 'classroom', 'online', 'distance', 'scholarship',
                'tuition', 'admission', 'exam', 'test', 'grade'
            ]
        }

    def classify_topic(self, text: str, title: str = "") -> Dict[str, Any]:
        """
        Classify the topic of a news article

        Args:
            text: Article content
            title: Article title

        Returns:
            Dictionary with primary topic, secondary topics, and confidence scores
        """
        try:
            # Combine title and text for better classification
            combined_text = f"{title} {text}".lower()

            # Remove punctuation and tokenize
            clean_text = re.sub(r'[^\w\s]', ' ', combined_text)
            words = word_tokenize(clean_text)

            # Remove stop words
            filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]

            # Calculate topic scores
            topic_scores = {}
            for topic, keywords in self.topic_keywords.items():
                score = sum(1 for word in filtered_words if word in keywords)
                if score > 0:
                    topic_scores[topic] = score

            if not topic_scores:
                return {
                    'primary': 'General',
                    'secondary': [],
                    'confidence': 0.0
                }

            # Sort topics by score
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

            # Primary topic
            primary_topic = sorted_topics[0][0]
            primary_score = sorted_topics[0][1]

            # Secondary topics (topics with score > 30% of primary)
            secondary_topics = []
            threshold = primary_score * 0.3
            for topic, score in sorted_topics[1:]:
                if score >= threshold and len(secondary_topics) < 2:
                    secondary_topics.append(topic)

            # Calculate confidence based on score relative to text length
            text_length = len(filtered_words)
            confidence = min(primary_score / max(text_length * 0.1, 1), 1.0)

            return {
                'primary': primary_topic,
                'secondary': secondary_topics,
                'confidence': round(confidence, 3)
            }

        except Exception as e:
            logger.error(f"Error classifying topic: {e}")
            return {
                'primary': 'General',
                'secondary': [],
                'confidence': 0.0
            }


class KeywordExtractor:
    """Keyword extraction for news articles"""

    def __init__(self):
        self.stop_words = set()

        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK data not found. Installing...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def extract_keywords(self, text: str, title: str = "", num_keywords: int = 10) -> List[str]:
        """
        Extract keywords from article text

        Args:
            text: Article content
            title: Article title
            num_keywords: Number of keywords to extract

        Returns:
            List of extracted keywords
        """
        try:
            # Combine title and text
            combined_text = f"{title} {text}".lower()

            # Remove punctuation and tokenize
            clean_text = re.sub(r'[^\w\s]', ' ', combined_text)
            words = word_tokenize(clean_text)

            # Remove stop words and short words
            filtered_words = [
                word for word in words
                if word not in self.stop_words and len(word) > 2
            ]

            # Count word frequencies
            word_freq = Counter(filtered_words)

            # Extract most common words as keywords
            keywords = [word for word, _ in word_freq.most_common(num_keywords)]

            return keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
