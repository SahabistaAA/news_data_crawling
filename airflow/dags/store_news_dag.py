from airflow import DAG
from airflow.providers.standard.operators import PythonOperator
from datetime import datetime, timedelta
from plugins.kafka_operator import KafkaMessageSensor

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}

def process_message(**kwargs):
    msgs = kwargs['ti'].xcom_pull(task_ids="wait_for_kafka", key="kafka_message")
    # process here (store into Postgres etc.)
    print(f"Got message: {msgs}")

with DAG(
    "consume_news_dag",
    default_args=default_args,
    description="Consume news from Kafka using sensor",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["news", "kafka"],
) as dag:

    wait_for_kafka = KafkaMessageSensor(
        task_id="wait_for_kafka",
        topic="news_topic",
        group_id="news_group",
        poke_interval=15,
        timeout=300,
    )

    consume_news = PythonOperator(
        task_id="consume_news",
        python_callable=process_message,
        provide_context=True,
    )

    wait_for_kafka >> consume_news
