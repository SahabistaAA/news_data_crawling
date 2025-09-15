#!/bin/bash
set -e

echo "Initializing Elasticsearch..."

# Wait until Elasticsearch is up
until curl -s http://localhost:9200 >/dev/null; do
    echo "Waiting for Elasticsearch..."
    sleep 2
done

# Create index
curl -X PUT "http://elasticsearch:9200/news_index" \
    -H 'Content-Type: application/json' \
    s-d @/configs/elasticsearch_index.json


echo "Elasticsearch initialized successfully."
