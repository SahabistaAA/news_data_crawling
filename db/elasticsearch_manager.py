from elasticsearch import Elasticsearch, helpers
from datetime import datetime, timezone
import json
import logging
from typing import List, Dict, Optional
import hashlib

logger = logging.getLogger(__name__)

class ElasticsearchManager:
    def __init__(self, host: str = "localhost", port: int = 9200, username: str = None, password: str = None, use_ssl: bool = False, ca_certs: str = None):
        self.conn = None
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.ca_certs = ca_certs
        self.connect()
        
    def connect(self):
        """Connect to Elasticsearch instance."""
        try:
            scheme = "https" if self.use_ssl else "http"
            
            es_config  = {
                "hosts" : [f"{scheme}://{self.host}:{self.port}"],
                "request_timeout" : 30,
                "verify_certs" : self.use_ssl,
            }
            
            if self.username and self.password:
                es_config["basic_auth"] = (self.username, self.password)
                
            if self.use_ssl and self.ca_certs:
                es_config["ca_certs"] = self.ca_certs
                
            self.conn = Elasticsearch(**es_config)

            if self.conn.ping():
                logger.info("Connected to Elasticsearch successfully.")
            else:
                logger.error("Could not connect to Elasticsearch.")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.conn = None
    
    def close(self):
        """Close the Elasticsearch connection."""
        if self.conn:
            self.conn.close()
            logger.info("Elasticsearch connection closed.")
            
    def create_index(self, index_name: str, mapping: Dict = None):
        """Create an Elasticsearch index with optional mappings."""
        if not self.conn:
            logger.error("Elasticsearch connection not established.")
            return False
        
        try:
            if self.conn.indices(index = index_name):
                logger.info(f"Index '{index_name}' already exists.")
                return True
            
            create_body = mapping or {}
            self.conn.indices.create(index=index_name, body=create_body, ignore=404)
            logger.info(f"Index '{index_name}' created successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            
    def _generate_doc_id(self, document: Dict) -> str:
        """Generate consistent document ID based on URL and title or content."""
        if document.get("url") and document.get("title"):
            content_str = f"{document['url']}_{document['title']}"
            return hashlib.md5(content_str.encode()).hexdigest()
        
        # Fallback to content hash if URL or title is missing
        doc_str = json.dumps(document, sort_keys=True)
        return hashlib.md5(doc_str.encode()).hexdigest()
    
    def index_document(self, index_name: str, document: Dict, doc_id: Optional[str] = None) -> bool:
        """Index a single document"""
        if not self.conn:
            return False
        
        try:
            if "crawled_date" not in document:
                document["crawled_date"] = datetime.now(timezone.utc).isoformat()
            
            if "processed_date" not in document:
                document["processed_date"] = datetime.now(timezone.utc).isoformat()
                
            response = self.conn.index(
                index =  index_name,
                id = doc_id or self._generate_doc_id(document),
                document = document,
            )
            
            if response.get("result") in ["created", "updated"]:
                logger.info(f"Document indexed successfully in '{index_name}' with ID: {response['_id']}")
                return True
            else:
                logger.error(f"Failed to index document in '{index_name}': {response}")
                return False
        except Exception as e:
            logger.error(f"Error indexing document in '{index_name}': {e}")
            return False
        
    def bulk_index(self, index_name: str, documents: List[Dict]) -> bool:
        """Bulk index multiple documents"""
        if not self.conn:
            return False
        
        try:
            actions = []
            
            for doc in documents:
                if "crawled_date" not in doc:
                    doc["crawled_date"] = datetime.now(timezone.utc).isoformat()
                if "processed_date" not in doc:
                    doc["processed_date"] = datetime.now(timezone.utc).isoformat()
                
                doc_id = self._generate_doc_id(doc)

                action = {
                    "_op_type" : "index",
                    "_index" : index_name,
                    "_id" : doc_id,
                    "_source": doc
                }
                actions.append(action)
            
            try:
                success, errors = helpers.bulk(
                    self.conn,
                    actions,
                    stats_only=True,
                    raise_on_error=False,
                    raise_on_exception=False
                )
                
                if errors:
                    logger.warning(f"Some documents are failed to index: {len(errors)} errors.")
                    for error in errors[:5]:
                        logger.warning(f"Bulk error: {error}")
                
                logger.info(f"Bulk indexed {success} documents into '{index_name}'.")
                return success
            except Exception as bulk_e:
                logger.error(f"Bulk indexing error: {bulk_e}")
                return False
            
        except Exception as e:
            logger.error(f"Error preparing bulk index actions: {e}")
            return False
        
    def search_document(self, index_name: str, query: Dict, size: int = 10) -> List[Dict]:
        """Search documents in Elasticsearch index."""
        if not self.conn:
            return []
        
        try:
            response = self.conn.search(
                index = index_name,
                body = query,
                size = size
            )
            
            return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]
        except Exception as e:
            logger.error(f"Error searching documents in '{index_name}': {e}")
            return []
    
    def get_index_stats(self, index_name: str) -> Optional[Dict]:
        """Get statistics of an Elasticsearch index."""
        if not self.conn:
            return None
        
        try:
            stats = self.conn.indices.stats(index=index_name)
            return stats
        except Exception as e:
            logger.error(f"Error getting stats for index '{index_name}': {e}")
            return None
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an Elasticsearch index."""
        if not self.conn:
            return False
        
        try:
            self.conn.indices.delete(index=index_name, ignore=[400, 404])
            logger.info(f"Index '{index_name}' deleted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error deleting index '{index_name}': {e}")
            return False
        
    def create_news_index(self, mapping: Dict, index_name : str = "news_index"):
        mapping = mapping or {
            "settings" : {
                "analysis" : {
                    "filter" : {
                        "indonesian_stop" : {
                            "type" : "stop",
                            "stopwords" : "_indonesian"
                        },
                        "indonesian_stemmer" : {
                            "type" : "stemmer",
                            "language" : "indonesian"
                        },
                        "english_stop" : {
                            "type" : "stop",
                            "stopwords" : "english"
                        },
                        "english_stemmer" : {
                            "type" : "stemmer",
                            "language" : "english"
                        }
                    },
                    "analyzer" :{
                        "indonesian_custom" : {
                            "tokenizer" : "standard",
                            "filter" : ["lowercase", "indonesian_stop", "indonesian_stemmer"]
                        },
                        "english_custom" : {
                            "tokenizer" : "standard",
                            "filter" : ["lowercase", "english_stop", "english_stemmer"]
                        }
                    }
                },
                "number_of_shards" : 1,
                "number_of_replicas" : 0
            },
            "mappings" : {
                "id" : {
                    "type" : "keyword"
                },
                "title" : {
                    "type" : "text",
                    "analyzer" : "english_custom",
                    "fields" : {
                        "indonesian" : {
                            "type" : "text",
                            "analyzer" : "indonesian_custom"
                        },
                        "keywords" : {
                            "type" : "keyword"
                        }
                    }
                },
                "content" : {
                    "type" : "text",
                    "analyzer" : "english_custom",
                    "fields" : {
                        "indonesian" : {
                            "type" : "text",
                            "analyzer" : "indonesian_custom"
                        }
                    }
                },
                "full_content" : {
                    "type" : "text",
                    "analyzer" : "english_custom",
                    "fields" : {
                        "indonesian" : {
                            "type" : "text",
                            "analyzer" : "indonesian_custom"
                        }
                    }
                },
                "source" : {
                    "type" : "keyword"
                },
                "category" : {
                    "type" : "keyword"
                },
                "published_date": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ssZZ"
                },
                "crawled_date" : {
                    "type" : "date",
                    "format" : "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ssZZ"
                },
                "processed_date" : {
                    "type" : "date",
                    "format" : "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ssZZ"
                },
                "url" : {
                    "type" : "keyword"
                },
                "language" : {
                    "type" : "keyword"
                },
                "region" : {
                    "type" : "keyword"
                },
                "country" : {
                    "type" : "keyword"
                },
                "tags" : {
                    "type" : "keyword"
                },
                "source_type" : {
                    "type" : "keyword"
                },
                "author" : {
                    "type" : "keyword"
                }
            }
        }
        
        return self.create_index(index_name=index_name, mapping=mapping)