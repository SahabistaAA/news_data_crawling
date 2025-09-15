from airflow import DAG
from airflow.providers.standard.operators import PythonOperator
from airflow.sensors.base import PokeReturnValue
from airflow.providers.standard.sensors import PythonSensor
from datetime import datetime, timedelta
from plugins.kafka_operator import KafkaProducerOperator
from crawlers.news_crawler import NewsCrawler

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}

def crawl_rss_task(**kwargs):
    crawler = NewsCrawler()
    articles = crawler.crawl_rss()
    kwargs['ti'].xcom_push(key="raw_articles", value=articles)

def check_crawled_articles(**kwargs) -> PokeReturnValue:
    articles = kwargs['ti'].xcom_pull(dag_id="crawl_news_dag", task_ids="crawl_news", key="raw_articles")
    if articles:
        return PokeReturnValue(is_done=True, xcom_value=articles)
    return PokeReturnValue(is_done=False)

def produce_news_task(articles, **kwargs):
    producer = KafkaProducerOperator(topic="news_topic")
    producer.produce_messages(articles)

with DAG(
    "produce_news_dag",
    default_args=default_args,
    description="Produce crawled news into Kafka with sensor waiting for crawler",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["news", "kafka"],
) as dag:

    wait_for_crawl = PythonSensor(
        task_id="wait_for_crawl",
        python_callable=check_crawled_articles,
        poke_interval=30,
        timeout=300,
        mode="poke",
        provide_context=True,
    )

    produce_news = PythonOperator(
        task_id="produce_news",
        python_callable=produce_news_task,
        op_args=["{{ ti.xcom_pull(task_ids='wait_for_crawl') }}"],
        provide_context=True,
    )

    wait_for_crawl >> produce_news
