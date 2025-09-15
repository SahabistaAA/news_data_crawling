from neo4j import GraphDatabase
import logging
from typing import Dict, List, Optional

class Neo4jManager:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.driver = None
        self.uri = uri
        self.user = user
        self.password = password
        self.connect()

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logging.info("Connected to Neo4j successfully.")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed.")

    def run_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        if not self.driver:
            logging.error("Neo4j driver not initialized.")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logging.error(f"Error running query: {e}")
            return []

    def create_entity(self, label: str, key: str, value: str) -> Dict:
        """Create a node of type label (e.g., Person, Organization, Topic)."""
        query = f"""
        MERGE (n:{label} {{{key}: $value}})
        RETURN n
        """
        return self.run_query(query, {"value": value})

    def create_article_relation(self, node1_label: str, node1_key: str, node1_value: str,
                                node2_label: str, node2_key: str, node2_value: str,
                                article_props: Dict):
        """
        Create an ARTICLE relationship between two nodes with metadata.
        Example: (Person)-[:ARTICLE {title, url, pub_date}]->(Organization)
        """
        query = f"""
            MERGE (a:{node1_label} {{{node1_key}: $value1}})
            MERGE (b:{node2_label} {{{node2_key}: $value2}})
            MERGE (a)-[r:ARTICLE {{url: $url}}]->(b)
            SET r += $props
            RETURN r
        """
        return self.run_query(query, 
            {
                "value1": node1_value,
                "value2": node2_value,
                "url": article_props.get("url"),
                "props": article_props
            }
        )
