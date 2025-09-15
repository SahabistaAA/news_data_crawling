import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
from typing import Dict, Any, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download issue: {e}")


class SentimentAnalyzer:
    """Advanced sentiment analysis for multiple languages"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.vader_analyzer = None
        self.models_loaded = False
    
    def load_models(self) -> None:
        """Load all sentiment analysis models"""
        try:
            logger.info("Loading sentiment analysis models...")
            
            # Load Indonesian models
            self._load_indonesian_models()
            
            # Load English models
            self._load_english_models()
            
            # Load multilingual models
            self._load_multilingual_models()
            
            # Initialize VADER for English
            try:
                self.vader_analyzer = VaderAnalyzer()
                logger.info("VADER sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"VADER loading issue: {e}")
            
            self.models_loaded = True
            logger.info("All sentiment models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment models: {e}")
            raise
    
    def _load_indonesian_models(self) -> None:
        """Load Indonesian sentiment models"""
        try:
            # IndoBERT for Indonesian sentiment
            model_name = "indobenchmark/indobert-base-p1"
            
            self.tokenizers['indonesian'] = AutoTokenizer.from_pretrained(model_name)
            self.models['indonesian'] = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=3  # negative, neutral, positive
            )
            
            logger.info("Indonesian sentiment models loaded")
            
        except Exception as e:
            logger.warning(f"Could not load Indonesian models: {e}")
            # Fallback to multilingual model
            self._load_multilingual_fallback('indonesian')
    
    def _load_english_models(self) -> None:
        """Load English sentiment models"""
        try:
            # RoBERTa-based English sentiment
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            self.pipelines['english'] = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("English sentiment models loaded")
            
        except Exception as e:
            logger.warning(f"Could not load English models: {e}")
            # Fallback to TextBlob
            self.pipelines['english'] = None
    
    def _load_multilingual_models(self) -> None:
        """Load multilingual sentiment models"""
        try:
            # XLM-RoBERTa for multilingual sentiment
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            
            self.pipelines['multilingual'] = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Multilingual sentiment models loaded")
            
        except Exception as e:
            logger.warning(f"Could not load multilingual models: {e}")
    
    def _load_multilingual_fallback(self, language: str) -> None:
        """Load multilingual model as fallback"""
        try:
            if 'multilingual' not in self.pipelines:
                self._load_multilingual_models()
        except Exception as e:
            logger.error(f"Fallback model loading failed: {e}")
    
    def analyze_sentiment(self, text: str, language: str = 'english') -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            language: Language of the text ('english', 'indonesian', etc.)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if not self.models_loaded:
                self.load_models()
            
            # Normalize language parameter
            language = language.lower().strip()
            if language in ['id', 'indonesian']:
                language = 'indonesian'
            elif language in ['en', 'english']:
                language = 'english'
            else:
                language = 'multilingual'
            
            # Analyze based on language
            if language == 'indonesian':
                return self._analyze_indonesian_sentiment(text)
            elif language == 'english':
                return self._analyze_english_sentiment(text)
            else:
                return self._analyze_multilingual_sentiment(text)
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._fallback_sentiment(text)
    
    def _analyze_indonesian_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze Indonesian text sentiment"""
        try:
            if 'indonesian' not in self.tokenizers:
                # Fall back to multilingual
                return self._analyze_multilingual_sentiment(text)
            
            tokenizer = self.tokenizers['indonesian']
            model = self.models['indonesian']
            
            # Tokenize and encode
            inputs = tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                scores = predictions[0].tolist()
            
            # Map to sentiment labels
            labels = ['negative', 'neutral', 'positive']
            max_idx = np.argmax(scores)
            sentiment_label = labels[max_idx]
            confidence = scores[max_idx]
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = scores[2] - scores[0]  # positive - negative
            
            return {
                'score': float(sentiment_score),
                'label': sentiment_label,
                'confidence': float(confidence),
                'emotion_scores': {
                    'negative': float(scores[0]),
                    'neutral': float(scores[1]),
                    'positive': float(scores[2])
                },
                'model_used': 'indobert'
            }
            
        except Exception as e:
            logger.error(f"Indonesian sentiment analysis error: {e}")
            return self._analyze_multilingual_sentiment(text)
    
    def _analyze_english_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze English text sentiment"""
        try:
            if self.pipelines.get('english') is not None:
                # Use RoBERTa model
                result = self.pipelines['english'](text)[0]
                
                # Map labels to standard format
                label_mapping = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral', 
                    'LABEL_2': 'positive',
                    'NEGATIVE': 'negative',
                    'NEUTRAL': 'neutral',
                    'POSITIVE': 'positive'
                }
                
                label = label_mapping.get(result['label'].upper(), result['label'].lower())
                confidence = result['score']
                
                # Calculate sentiment score
                if label == 'positive':
                    score = confidence
                elif label == 'negative':
                    score = -confidence
                else:
                    score = 0.0
                
                # Get VADER scores for emotion breakdown
                emotion_scores = {}
                if self.vader_analyzer:
                    try:
                        vader_scores = self.vader_analyzer.polarity_scores(text)
                        emotion_scores = {
                            'negative': vader_scores['neg'],
                            'neutral': vader_scores['neu'],
                            'positive': vader_scores['pos'],
                            'compound': vader_scores['compound']
                        }
                    except Exception as e:
                        logger.warning(f"VADER analysis failed: {e}")
                
                return {
                    'score': float(score),
                    'label': label,
                    'confidence': float(confidence),
                    'emotion_scores': emotion_scores,
                    'model_used': 'roberta'
                }
            
            else:
                # Fallback to VADER only
                return self._analyze_with_vader(text)
                
        except Exception as e:
            logger.error(f"English sentiment analysis error: {e}")
            return self._analyze_with_vader(text)
    
    def _analyze_multilingual_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text with multilingual model"""
        try:
            if 'multilingual' not in self.pipelines or self.pipelines['multilingual'] is None:
                return self._fallback_sentiment(text)
            
            result = self.pipelines['multilingual'](text)[0]
            
            # Map labels
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            label = label_mapping.get(result['label'].upper(), result['label'].lower())
            confidence = result['score']
            
            # Calculate sentiment score
            if label == 'positive':
                score = confidence
            elif label == 'negative':
                score = -confidence
            else:
                score = 0.0
            
            return {
                'score': float(score),
                'label': label,
                'confidence': float(confidence),
                'emotion_scores': {
                    'confidence': confidence
                },
                'model_used': 'xlm-roberta'
            }
            
        except Exception as e:
            logger.error(f"Multilingual sentiment analysis error: {e}")
            return self._fallback_sentiment(text)
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER"""
        try:
            if not self.vader_analyzer:
                return self._fallback_sentiment(text)
            
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Determine label based on compound score
            if compound >= 0.05:
                label = 'positive'
                confidence = scores['pos']
                score = compound
            elif compound <= -0.05:
                label = 'negative'
                confidence = scores['neg']
                score = compound
            else:
                label = 'neutral'
                confidence = scores['neu']
                score = 0.0
            
            return {
                'score': float(score),
                'label': label,
                'confidence': float(confidence),
                'emotion_scores': {
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'positive': scores['pos'],
                    'compound': scores['compound']
                },
                'model_used': 'vader'
            }
            
        except Exception as e:
            logger.error(f"VADER sentiment analysis error: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                label = 'positive'
                confidence = min(abs(polarity) + 0.5, 1.0)
            elif polarity < -0.1:
                label = 'negative'
                confidence = min(abs(polarity) + 0.5, 1.0)
            else:
                label = 'neutral'
                confidence = 0.6
            
            return {
                'score': float(polarity),
                'label': label,
                'confidence': float(confidence),
                'emotion_scores': {
                    'polarity': polarity,
                    'subjectivity': subjectivity
                },
                'model_used': 'textblob'
            }
            
        except Exception as e:
            logger.error(f"Fallback sentiment analysis error: {e}")
            return {
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'emotion_scores': {},
                'model_used': 'fallback'
            }
    
    def batch_analyze_sentiment(self, texts: list, language: str = 'english') -> list:
        """Analyze sentiment for a batch of texts"""
        try:
            results = []
            for text in texts:
                result = self.analyze_sentiment(text, language)
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return [self._fallback_sentiment(text) for text in texts]