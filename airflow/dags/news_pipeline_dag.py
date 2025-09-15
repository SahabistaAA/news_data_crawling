from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    "news_pipeline_dag",
    default_args=default_args,
    description="Full news pipeline DAG",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["news", "pipeline"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    with TaskGroup("crawl") as crawl:
        from crawl_news_dag import crawl_news

    with TaskGroup("produce") as produce:
        from produce_news_dag import produce_news

    with TaskGroup("consume") as consume:
        from consume_news_dag import consume_news

    with TaskGroup("store") as store:
        from store_news_dag import store_news

    start >> crawl >> produce >> consume >> store >> end
