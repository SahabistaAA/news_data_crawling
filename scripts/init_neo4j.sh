#!/bin/bash
set -e

echo "Initializing Neo4j..."

# Wait until Neo4j is available
until curl -s http://localhost:7474 >/dev/null; do
  echo "Waiting for Neo4j..."
  sleep 2
done

# Create constraints for nodes
cypher-shell -u neo4j -p password <<EOF
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.url IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Publisher) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyphrase) REQUIRE k.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (per:Person) REQUIRE per.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (org:Organization) REQUIRE org.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (pl:Place) REQUIRE pl.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (co:Concept) REQUIRE co.name IS UNIQUE;
EOF

echo "Neo4j initialized successfully."