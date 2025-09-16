# TODO: Create Full News Data Pipeline

## 1. Fix store_news_dag.py
- Correct DAG name from "consume_news_dag" to "store_news_dag"
- Implement proper storage logic: save raw data to PostgreSQL, mapped data to Elasticsearch
- Use db managers for storage

## 2. Implement bronze_processing_dag.py
- Create DAG with tasks for bronze layer processing
- Tasks: cleaning, normalization, standardization
- Use medallion_architect/bronze_layer/ modules
- Read from raw schema in PostgreSQL, save to bronze schema

## 3. Implement silver_enrichment_dag.py
- Create DAG with tasks for silver layer enrichment
- Tasks: NER, Sentiment Analysis, Knowledge Graph
- Save NER and Sentiment to PostgreSQL silver schema
- Save Knowledge Graph to Neo4j
- Use medallion_architect/silver_layer/ modules

## 4. Implement gold_aggregation_dag.py
- Create DAG with tasks for gold layer aggregation
- Tasks: aggregation, analytics
- Use medallion_architect/gold_layer/ modules
- Save to gold schema in PostgreSQL

## 5. Update news_pipeline_dag.py
- Add TaskGroups for bronze, silver, gold after store
- Ensure proper sequencing: crawl >> produce >> consume >> store >> bronze >> silver >> gold

## 6. Test and Run
- Run docker-compose to start all services
- Test the full pipeline
