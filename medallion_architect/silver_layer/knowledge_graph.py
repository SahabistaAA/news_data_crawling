import logging
from neo4j import GraphDatabase
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Build and manage knowledge graph in Neo4j"""
    
    def __init__(self):
        self.driver = None
        self.neo4j_config = {
            'uri': os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'neo4j_password')
        }
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def create_indexes(self):
        """Create indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.url)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (o:Organization) ON (o.name)",
            "CREATE INDEX IF NOT EXISTS FOR (l:Place) ON (l.name)",
            "CREATE INDEX IF NOT EXISTS FOR (pub:Publisher) ON (pub.name)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Category) ON (c.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (k:Keyphrase) ON (k.phrase)",
            "CREATE INDEX IF NOT EXISTS FOR (con:Concept) ON (con.name)"
        ]
        
        try:
            with self.driver.session() as session:
                for index_query in indexes:
                    session.run(index_query)
            logger.info("Neo4j indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise
    
    def create_knowledge_graph(self, articles_df: pd.DataFrame):
        """Create knowledge graph from enriched articles data"""
        try:
            logger.info(f"Creating knowledge graph for {len(articles_df)} articles")
            
            # Create indexes first
            self.create_indexes()
            
            # Process articles in batches
            batch_size = 50
            for i in range(0, len(articles_df), batch_size):
                batch = articles_df.iloc[i:i + batch_size]
                self._process_batch(batch)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(articles_df)-1)//batch_size + 1}")
            
            logger.info("Knowledge graph creation completed")
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {e}")
            raise
    
    def _process_batch(self, batch: pd.DataFrame):
        """Process a batch of articles"""
        try:
            with self.driver.session() as session:
                for _, row in batch.iterrows():
                    self._create_article_graph(session, row)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    def _create_article_graph(self, session, article_row):
        """Create graph for a single article"""
        try:
            # Create or update Article node
            article_id = self._create_article_node(session, article_row)
            
            # Create Publisher node and relationship
            self._create_publisher_relationship(session, article_id, article_row)
            
            # Create Category/Topic nodes and relationships
            self._create_topic_relationships(session, article_id, article_row)
            
            # Create Person entities and relationships
            self._create_person_relationships(session, article_id, article_row)
            
            # Create Organization entities and relationships
            self._create_organization_relationships(session, article_id, article_row)
            
            # Create Place entities and relationships
            self._create_place_relationships(session, article_id, article_row)
            
            # Create Concept and Keyphrase relationships
            self._create_concept_relationships(session, article_id, article_row)
            
        except Exception as e:
            logger.error(f"Error creating article graph: {e}")
            raise
    
    def _create_article_node(self, session, row) -> str:
        """Create or update Article node"""
        try:
            article_query = """
            MERGE (a:Article {url: $url})
            SET a.title = $title,
                a.pub_date = datetime($pub_date),
                a.content = $content,
                a.sentiment_score = $sentiment_score,
                a.sentiment_label = $sentiment_label,
                a.sentiment_confidence = $sentiment_confidence,
                a.language = $language,
                a.quality_score = $quality_score,
                a.content_type = $content_type,
                a.updated_at = datetime()
            RETURN a.url as article_id
            """
            
            # Prepare parameters
            params = {
                'url': str(row.get('url', '')),
                'title': str(row.get('title', '')),
                'pub_date': row.get('pub_date').isoformat() if pd.notna(row.get('pub_date')) else datetime.now().isoformat(),
                'content': str(row.get('content', ''))[:1000],  # Truncate content
                'sentiment_score': float(row.get('sentiment_score', 0.0)),
                'sentiment_label': str(row.get('sentiment_label', 'neutral')),
                'sentiment_confidence': float(row.get('sentiment_confidence', 0.0)),
                'language': str(row.get('language_detected', 'unknown')),
                'quality_score': float(row.get('quality_score', 0.0)),
                'content_type': str(row.get('content_type', 'news'))
            }
            
            result = session.run(article_query, params)
            article_id = result.single()['article_id']
            return article_id
            
        except Exception as e:
            logger.error(f"Error creating article node: {e}")
            raise
    
    def _create_publisher_relationship(self, session, article_id: str, row):
        """Create Publisher node and relationship"""
        try:
            source = str(row.get('source', 'Unknown'))
            region = str(row.get('region', 'Unknown'))
            author = str(row.get('author', ''))
            
            if source and source != 'Unknown':
                publisher_query = """
                MATCH (a:Article {url: $article_id})
                MERGE (p:Publisher {name: $source})
                SET p.region = $region,
                    p.updated_at = datetime()
                MERGE (a)-[r:PUBLISHED_BY]->(p)
                SET r.created_at = datetime()
                """
                
                session.run(publisher_query, {
                    'article_id': article_id,
                    'source': source,
                    'region': region
                })
            
            # Create author relationship if author exists
            if author and author not in ['', 'nan', 'Unknown']:
                author_query = """
                MATCH (a:Article {url: $article_id})
                MERGE (auth:Person {name: $author})
                SET auth.type = 'author',
                    auth.updated_at = datetime()
                MERGE (a)-[r:AUTHORED_BY]->(auth)
                SET r.created_at = datetime()
                """
                
                session.run(author_query, {
                    'article_id': article_id,
                    'author': author
                })
                
        except Exception as e:
            logger.error(f"Error creating publisher relationship: {e}")
    
    def _create_topic_relationships(self, session, article_id: str, row):
        """Create Topic/Category relationships"""
        try:
            content_type = str(row.get('content_type', 'news'))
            
            # Create Category node
            category_query = """
            MATCH (a:Article {url: $article_id})
            MERGE (c:Category {name: $category})
            SET c.updated_at = datetime()
            MERGE (a)-[r:BELONGS_TO_CATEGORY]->(c)
            SET r.created_at = datetime()
            """
            
            session.run(category_query, {
                'article_id': article_id,
                'category': content_type
            })
            
            # Create Topic based on content analysis
            topic = self._extract_main_topic(row)
            if topic:
                topic_query = """
                MATCH (a:Article {url: $article_id})
                MERGE (t:Topic {name: $topic})
                SET t.updated_at = datetime()
                MERGE (a)-[r:DISCUSSES_TOPIC]->(t)
                SET r.relevance_score = $relevance,
                    r.created_at = datetime()
                """
                
                session.run(topic_query, {
                    'article_id': article_id,
                    'topic': topic,
                    'relevance': 0.8  # Default relevance score
                })
                
        except Exception as e:
            logger.error(f"Error creating topic relationships: {e}")
    
    def _create_person_relationships(self, session, article_id: str, row):
        """Create Person entity relationships"""
        try:
            person_entities = row.get('person_entities', [])
            if isinstance(person_entities, str):
                person_entities = json.loads(person_entities) if person_entities else []
            
            for person in person_entities[:10]:  # Limit to 10 persons
                if person and len(person.strip()) > 1:
                    person_query = """
                    MATCH (a:Article {url: $article_id})
                    MERGE (p:Person {name: $person})
                    SET p.type = 'entity',
                        p.updated_at = datetime()
                    MERGE (a)-[r:MENTIONS_PERSON]->(p)
                    SET r.created_at = datetime()
                    """
                    
                    session.run(person_query, {
                        'article_id': article_id,
                        'person': person.strip()
                    })
                    
        except Exception as e:
            logger.error(f"Error creating person relationships: {e}")
    
    def _create_organization_relationships(self, session, article_id: str, row):
        """Create Organization entity relationships"""
        try:
            org_entities = row.get('organization_entities', [])
            if isinstance(org_entities, str):
                org_entities = json.loads(org_entities) if org_entities else []
            
            for org in org_entities[:10]:  # Limit to 10 organizations
                if org and len(org.strip()) > 1:
                    org_query = """
                    MATCH (a:Article {url: $article_id})
                    MERGE (o:Organization {name: $org})
                    SET o.updated_at = datetime()
                    MERGE (a)-[r:MENTIONS_ORGANIZATION]->(o)
                    SET r.created_at = datetime()
                    """
                    
                    session.run(org_query, {
                        'article_id': article_id,
                        'org': org.strip()
                    })
                    
        except Exception as e:
            logger.error(f"Error creating organization relationships: {e}")
    
    def _create_place_relationships(self, session, article_id: str, row):
        """Create Place entity relationships"""
        try:
            location_entities = row.get('location_entities', [])
            if isinstance(location_entities, str):
                location_entities = json.loads(location_entities) if location_entities else []
            
            for location in location_entities[:10]:  # Limit to 10 locations
                if location and len(location.strip()) > 1:
                    place_query = """
                    MATCH (a:Article {url: $article_id})
                    MERGE (l:Place {name: $location})
                    SET l.updated_at = datetime()
                    MERGE (a)-[r:MENTIONS_PLACE]->(l)
                    SET r.created_at = datetime()
                    """
                    
                    session.run(place_query, {
                        'article_id': article_id,
                        'location': location.strip()
                    })
                    
        except Exception as e:
            logger.error(f"Error creating place relationships: {e}")
    
    def _create_concept_relationships(self, session, article_id: str, row):
        """Create Concept and Keyphrase relationships"""
        try:
            # Extract key concepts from title and content
            concepts = self._extract_concepts(row)
            keyphrases = self._extract_keyphrases(row)
            
            # Create Concept nodes
            for concept in concepts[:5]:  # Limit to 5 concepts
                concept_query = """
                MATCH (a:Article {url: $article_id})
                MERGE (c:Concept {name: $concept})
                SET c.updated_at = datetime()
                MERGE (a)-[r:RELATED_TO_CONCEPT]->(c)
                SET r.created_at = datetime()
                """
                
                session.run(concept_query, {
                    'article_id': article_id,
                    'concept': concept
                })
            
            # Create Keyphrase nodes
            for phrase in keyphrases[:10]:  # Limit to 10 keyphrases
                keyphrase_query = """
                MATCH (a:Article {url: $article_id})
                MERGE (k:Keyphrase {phrase: $phrase})
                SET k.updated_at = datetime()
                MERGE (a)-[r:CONTAINS_KEYPHRASE]->(k)
                SET r.created_at = datetime()
                """
                
                session.run(keyphrase_query, {
                    'article_id': article_id,
                    'phrase': phrase
                })
                
        except Exception as e:
            logger.error(f"Error creating concept relationships: {e}")
    
    def _extract_main_topic(self, row) -> str:
        """Extract main topic from article"""
        try:
            title = str(row.get('title', ''))
            content_type = str(row.get('content_type', 'news'))
            
            # Simple topic extraction based on keywords
            title_lower = title.lower()
            
            # Define topic keywords
            topic_keywords = {
                'politics': ['politik', 'government', 'election', 'president', 'minister', 'parliament'],
                'economics': ['ekonomi', 'economy', 'business', 'finance', 'market', 'trade'],
                'technology': ['teknologi', 'technology', 'digital', 'internet', 'AI', 'startup'],
                'health': ['kesehatan', 'health', 'medical', 'hospital', 'covid', 'pandemic'],
                'sports': ['olahraga', 'sports', 'football', 'basketball', 'olympic'],
                'entertainment': ['hiburan', 'entertainment', 'movie', 'music', 'celebrity'],
                'education': ['pendidikan', 'education', 'school', 'university', 'student']
            }
            
            # Find matching topic
            for topic, keywords in topic_keywords.items():
                if any(keyword in title_lower for keyword in keywords):
                    return topic
            
            # Default to content type
            return content_type
            
        except Exception as e:
            logger.error(f"Error extracting main topic: {e}")
            return 'general'
    
    def _extract_concepts(self, row) -> List[str]:
        """Extract key concepts from article"""
        try:
            title = str(row.get('title', ''))
            
            # Simple concept extraction - use title words
            words = title.split()
            concepts = []
            
            for word in words:
                word = word.strip('.,!?";()[]{}').lower()
                if len(word) > 3 and word not in ['yang', 'dari', 'dengan', 'untuk', 'this', 'that', 'with', 'from']:
                    concepts.append(word.capitalize())
            
            return concepts[:5]
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []
    
    def _extract_keyphrases(self, row) -> List[str]:
        """Extract keyphrases from article"""
        try:
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))[:500]  # First 500 chars
            
            # Combine title and content
            text = f"{title} {content}"
            
            # Simple keyphrase extraction - use bigrams and trigrams
            words = text.split()
            keyphrases = []
            
            # Extract bigrams
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                phrase = phrase.strip('.,!?";()[]{}').lower()
                if len(phrase) > 5:
                    keyphrases.append(phrase.title())
            
            # Extract trigrams
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrase = phrase.strip('.,!?";()[]{}').lower()
                if len(phrase) > 8:
                    keyphrases.append(phrase.title())
            
            # Remove duplicates and return
            return list(dict.fromkeys(keyphrases))[:10]
            
        except Exception as e:
            logger.error(f"Error extracting keyphrases: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        try:
            stats_queries = {
                'articles': "MATCH (a:Article) RETURN count(a) as count",
                'persons': "MATCH (p:Person) RETURN count(p) as count",
                'organizations': "MATCH (o:Organization) RETURN count(o) as count",
                'places': "MATCH (l:Place) RETURN count(l) as count",
                'publishers': "MATCH (pub:Publisher) RETURN count(pub) as count",
                'topics': "MATCH (t:Topic) RETURN count(t) as count",
                'concepts': "MATCH (c:Concept) RETURN count(c) as count",
                'keyphrases': "MATCH (k:Keyphrase) RETURN count(k) as count",
                'relationships': "MATCH ()-[r]->() RETURN count(r) as count"
            }
            
            stats = {}
            with self.driver.session() as session:
                for key, query in stats_queries.items():
                    result = session.run(query)
                    stats[key] = result.single()['count']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def cleanup_old_nodes(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old nodes and relationships"""
        try:
            cleanup_query = """
            MATCH (a:Article)
            WHERE a.updated_at < datetime() - duration('P{days}D')
            DETACH DELETE a
            RETURN count(a) as deleted_articles
            """.format(days=days_to_keep)
            
            with self.driver.session() as session:
                result = session.run(cleanup_query)
                deleted_count = result.single()['deleted_articles']
                
                # Clean up orphaned nodes
                orphan_cleanup_queries = [
                    "MATCH (p:Person) WHERE NOT (p)--() DELETE p",
                    "MATCH (o:Organization) WHERE NOT (o)--() DELETE o",
                    "MATCH (l:Place) WHERE NOT (l)--() DELETE l",
                    "MATCH (c:Concept) WHERE NOT (c)--() DELETE c",
                    "MATCH (k:Keyphrase) WHERE NOT (k)--() DELETE k"
                ]
                
                for query in orphan_cleanup_queries:
                    session.run(query)
                
            return {'deleted_articles': deleted_count}
            
        except Exception as e:
            logger.error(f"Error cleaning up old nodes: {e}")
            return {'deleted_articles': 0}