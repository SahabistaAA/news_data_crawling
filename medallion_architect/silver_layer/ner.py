import logging
import spacy
from spacy.lang.id import Indonesian
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Advanced Named Entity Recognition for multiple languages"""
    
    def __init__(self):
        self.nlp_models = {}
        self.ner_pipelines = {}
        self.models_loaded = False
        
        # Entity type mappings
        self.entity_mappings = {
            'PERSON': 'person',
            'PER': 'person',
            'PEOPLE': 'person',
            'ORG': 'organization',
            'ORGANIZATION': 'organization',
            'CORP': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'LOCATION': 'location',
            'PLACE': 'location',
            'FAC': 'location',
            'FACILITY': 'location',
            'MISC': 'miscellaneous',
            'MISCELLANEOUS': 'miscellaneous',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'money',
            'PERCENT': 'percent',
            'QUANTITY': 'quantity'
        }
    
    def load_models(self) -> None:
        """Load all NER models"""
        try:
            logger.info("Loading NER models...")
            
            # Load spaCy models
            self._load_spacy_models()
            
            # Load transformer-based models
            self._load_transformer_models()
            
            self.models_loaded = True
            logger.info("All NER models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NER models: {e}")
            raise
    
    def _load_spacy_models(self) -> None:
        """Load spaCy NER models"""
        try:
            # English model
            try:
                self.nlp_models['english'] = spacy.load("en_core_web_sm")
                logger.info("English spaCy model loaded")
            except OSError:
                logger.warning("English spaCy model not found, trying alternative")
                try:
                    self.nlp_models['english'] = spacy.load("en_core_web_md")
                except OSError:
                    logger.warning("No English spaCy model available")
                    self.nlp_models['english'] = None
            
            # Indonesian model
            try:
                self.nlp_models['indonesian'] = spacy.load("id_core_news_sm")
                logger.info("Indonesian spaCy model loaded")
            except OSError:
                logger.warning("Indonesian spaCy model not found, using multilingual")
                try:
                    self.nlp_models['indonesian'] = spacy.load("xx_core_web_sm")
                except OSError:
                    logger.warning("No Indonesian spaCy model available")
                    self.nlp_models['indonesian'] = None
            
            # Multilingual fallback
            try:
                self.nlp_models['multilingual'] = spacy.load("xx_core_web_sm")
                logger.info("Multilingual spaCy model loaded")
            except OSError:
                logger.warning("Multilingual spaCy model not found")
                self.nlp_models['multilingual'] = None
                
        except Exception as e:
            logger.error(f"Error loading spaCy models: {e}")
    
    def _load_transformer_models(self) -> None:
        """Load transformer-based NER models"""
        try:
            # English NER with RoBERTa
            try:
                self.ner_pipelines['english'] = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("English transformer NER loaded")
            except Exception as e:
                logger.warning(f"Could not load English transformer NER: {e}")
                self.ner_pipelines['english'] = None
            
            # Multilingual NER
            try:
                self.ner_pipelines['multilingual'] = pipeline(
                    "ner",
                    model="Davlan/bert-base-multilingual-cased-ner-hrl",
                    tokenizer="Davlan/bert-base-multilingual-cased-ner-hrl",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Multilingual transformer NER loaded")
            except Exception as e:
                logger.warning(f"Could not load multilingual transformer NER: {e}")
                self.ner_pipelines['multilingual'] = None
                
        except Exception as e:
            logger.error(f"Error loading transformer models: {e}")
    
    def extract_entities(self, text: str, language: str = 'english') -> Dict[str, Any]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            language: Language of the text
            
        Returns:
            Dictionary with extracted entities
        """
        try:
            if not self.models_loaded:
                self.load_models()
            
            if not text or len(text.strip()) < 3:
                return self._empty_result()
            
            # Normalize language
            language = language.lower().strip()
            if language in ['id', 'indonesian']:
                language = 'indonesian'
            elif language in ['en', 'english']:
                language = 'english'
            else:
                language = 'multilingual'
            
            # Extract entities using multiple methods
            spacy_entities = self._extract_with_spacy(text, language)
            transformer_entities = self._extract_with_transformer(text, language)
            
            # Merge and deduplicate results
            merged_entities = self._merge_entity_results(spacy_entities, transformer_entities)
            
            # Post-process entities
            processed_entities = self._post_process_entities(merged_entities, text)
            
            # Organize by type
            return self._organize_entities(processed_entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return self._empty_result()
    
    def _extract_with_spacy(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy"""
        try:
            # Select appropriate model
            nlp = None
            if language in self.nlp_models and self.nlp_models[language] is not None:
                nlp = self.nlp_models[language]
            elif self.nlp_models.get('multilingual') is not None:
                nlp = self.nlp_models['multilingual']
            elif self.nlp_models.get('english') is not None:
                nlp = self.nlp_models['english']
            
            if nlp is None:
                return []
            
            # Process text
            doc = nlp(text[:1000000])  # Limit text length for performance
            
            entities = []
            for ent in doc.ents:
                entity_type = self.entity_mappings.get(ent.label_, 'miscellaneous')
                
                entities.append({
                    'text': ent.text.strip(),
                    'label': entity_type,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'score', 0.8),  # Default confidence
                    'source': 'spacy'
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
            return []
    
    def _extract_with_transformer(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract entities using transformer models"""
        try:
            # Select appropriate pipeline
            pipeline_key = language if language in self.ner_pipelines else 'multilingual'
            ner_pipeline = self.ner_pipelines.get(pipeline_key)
            
            if ner_pipeline is None:
                return []
            
            # Split text into chunks if too long
            max_length = 512
            chunks = self._split_text_into_chunks(text, max_length)
            
            all_entities = []
            char_offset = 0
            
            for chunk in chunks:
                try:
                    entities = ner_pipeline(chunk)
                    
                    for ent in entities:
                        entity_type = self.entity_mappings.get(
                            ent['entity_group'].upper(), 'miscellaneous'
                        )
                        
                        all_entities.append({
                            'text': ent['word'].strip(),
                            'label': entity_type,
                            'start': ent['start'] + char_offset,
                            'end': ent['end'] + char_offset,
                            'confidence': ent['score'],
                            'source': 'transformer'
                        })
                    
                    char_offset += len(chunk)
                    
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    char_offset += len(chunk)
                    continue
            
            return all_entities
            
        except Exception as e:
            logger.error(f"Error in transformer entity extraction: {e}")
            return []
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks for processing"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_entity_results(self, spacy_entities: List[Dict], 
                            transformer_entities: List[Dict]) -> List[Dict]:
        """Merge and deduplicate entity results from different sources"""
        try:
            # Combine all entities
            all_entities = spacy_entities + transformer_entities
            
            if not all_entities:
                return []
            
            # Sort by start position
            all_entities.sort(key=lambda x: (x['start'], x['end']))
            
            # Deduplicate overlapping entities
            merged_entities = []
            
            for entity in all_entities:
                # Check if this entity overlaps with any existing entity
                is_duplicate = False
                
                for existing in merged_entities:
                    if self._entities_overlap(entity, existing):
                        # Keep the entity with higher confidence
                        if entity['confidence'] > existing['confidence']:
                            merged_entities.remove(existing)
                            merged_entities.append(entity)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged_entities.append(entity)
            
            return merged_entities
            
        except Exception as e:
            logger.error(f"Error merging entity results: {e}")
            return spacy_entities + transformer_entities
    
    def _entities_overlap(self, ent1: Dict, ent2: Dict, threshold: float = 0.5) -> bool:
        """Check if two entities overlap significantly"""
        try:
            start1, end1 = ent1['start'], ent1['end']
            start2, end2 = ent2['start'], ent2['end']
            
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_end <= overlap_start:
                return False
            
            overlap_length = overlap_end - overlap_start
            min_length = min(end1 - start1, end2 - start2)
            
            if min_length == 0:
                return False
            
            overlap_ratio = overlap_length / min_length
            
            return overlap_ratio >= threshold
            
        except Exception as e:
            logger.error(f"Error checking entity overlap: {e}")
            return False
    
    def _post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """Post-process entities to improve quality"""
        try:
            processed = []
            
            for entity in entities:
                entity_text = entity['text'].strip()
                
                # Skip very short entities (less than 2 characters)
                if len(entity_text) < 2:
                    continue
                
                # Skip entities that are mostly punctuation
                if re.match(r'^[^\w\s]*$', entity_text):
                    continue
                
                # Skip common stop words and articles
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                if entity_text.lower() in stop_words:
                    continue
                
                # Clean up entity boundaries
                entity_text = self._clean_entity_text(entity_text)
                
                if entity_text:
                    entity_copy = entity.copy()
                    entity_copy['text'] = entity_text
                    processed.append(entity_copy)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error post-processing entities: {e}")
            return entities
    
    def _clean_entity_text(self, text: str) -> str:
        """Clean entity text"""
        try:
            # Remove leading/trailing punctuation except for meaningful ones
            text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Capitalize properly for names
            if text and len(text) > 1:
                # Simple capitalization for proper nouns
                words = text.split()
                cleaned_words = []
                for word in words:
                    if len(word) > 2:
                        cleaned_words.append(word.capitalize())
                    else:
                        cleaned_words.append(word)
                text = ' '.join(cleaned_words)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning entity text: {e}")
            return text
    
    def _organize_entities(self, entities: List[Dict]) -> Dict[str, Any]:
        """Organize entities by type"""
        try:
            result = {
                'entities': entities,
                'person_entities': [],
                'organization_entities': [],
                'location_entities': [],
                'misc_entities': [],
                'date_entities': [],
                'money_entities': []
            }
            
            for entity in entities:
                entity_type = entity['label']
                entity_text = entity['text']
                
                if entity_type == 'person':
                    result['person_entities'].append(entity_text)
                elif entity_type == 'organization':
                    result['organization_entities'].append(entity_text)
                elif entity_type == 'location':
                    result['location_entities'].append(entity_text)
                elif entity_type == 'date':
                    result['date_entities'].append(entity_text)
                elif entity_type == 'money':
                    result['money_entities'].append(entity_text)
                else:
                    result['misc_entities'].append(entity_text)
            
            # Remove duplicates while preserving order
            for key in ['person_entities', 'organization_entities', 'location_entities', 
                       'misc_entities', 'date_entities', 'money_entities']:
                result[key] = list(dict.fromkeys(result[key]))  # Remove duplicates
            
            # Add entity counts
            result['entity_counts'] = {
                'total': len(entities),
                'person': len(result['person_entities']),
                'organization': len(result['organization_entities']),
                'location': len(result['location_entities']),
                'date': len(result['date_entities']),
                'money': len(result['money_entities']),
                'misc': len(result['misc_entities'])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error organizing entities: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'entities': [],
            'person_entities': [],
            'organization_entities': [],
            'location_entities': [],
            'misc_entities': [],
            'date_entities': [],
            'money_entities': [],
            'entity_counts': {
                'total': 0,
                'person': 0,
                'organization': 0,
                'location': 0,
                'date': 0,
                'money': 0,
                'misc': 0
            }
        }
    
    def batch_extract_entities(self, texts: List[str], language: str = 'english') -> List[Dict[str, Any]]:
        """Extract entities from a batch of texts"""
        try:
            results = []
            for text in texts:
                result = self.extract_entities(text, language)
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"Error in batch entity extraction: {e}")
            return [self._empty_result() for _ in texts]
    
    def get_entity_statistics(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        try:
            total_texts = len(entities_list)
            total_entities = sum(len(ent_dict['entities']) for ent_dict in entities_list)
            
            # Count by type
            type_counts = {
                'person': sum(len(ent_dict['person_entities']) for ent_dict in entities_list),
                'organization': sum(len(ent_dict['organization_entities']) for ent_dict in entities_list),
                'location': sum(len(ent_dict['location_entities']) for ent_dict in entities_list),
                'misc': sum(len(ent_dict['misc_entities']) for ent_dict in entities_list),
                'date': sum(len(ent_dict.get('date_entities', [])) for ent_dict in entities_list),
                'money': sum(len(ent_dict.get('money_entities', [])) for ent_dict in entities_list)
            }
            
            # Calculate averages
            avg_entities_per_text = total_entities / total_texts if total_texts > 0 else 0
            
            return {
                'total_texts': total_texts,
                'total_entities': total_entities,
                'average_entities_per_text': round(avg_entities_per_text, 2),
                'entity_type_counts': type_counts,
                'entity_type_percentages': {
                    k: round((v / total_entities * 100), 2) if total_entities > 0 else 0
                    for k, v in type_counts.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating entity statistics: {e}")
            return {}