from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from crawlers.news_crawler import NewsCrawler

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def crawl_rss_task(**kwargs):
    crawler = NewsCrawler()
    articles = crawler.crawl_rss()
    kwargs['ti'].xcom_push(key="raw_articles", value=articles)

with DAG(
    "crawl_news_dag",
    default_args=default_args,
    description="Crawl raw news articles from RSS feeds",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["news", "crawl"],
) as dag:

    crawl_news = PythonOperator(
        task_id="crawl_news",
        python_callable=crawl_rss_task,
        provide_context=True,
    )
