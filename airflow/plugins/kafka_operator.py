from confluent_kafka import Producer, Consumer
import json, logging, os
from airflow.sensors.base import BaseSensorOperator

logger = logging.getLogger(__name__)

class KafkaProducerOperator:
    def __init__(self, topic: str):
        self.topic = topic
        self.producer = Producer({'bootstrap.servers': os.getenv("KAFKA_BROKER", "kafka:29092")})

    def produce_messages(self, articles):
        for article in articles:
            self.producer.produce(self.topic, json.dumps(article).encode("utf-8"))
        self.producer.flush()


class KafkaConsumerOperator:
    def __init__(self, topic: str, group_id: str):
        self.topic = topic
        self.consumer = Consumer({
            'bootstrap.servers': os.getenv("KAFKA_BROKER", "kafka:29092"),
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([self.topic])

    def consume_messages(self, max_messages=100):
        msgs = []
        for _ in range(max_messages):
            msg = self.consumer.poll(1.0)
            if msg is None: break
            if msg.error(): continue
            msgs.append(json.loads(msg.value().decode("utf-8")))
        self.consumer.close()
        return msgs
    
class KafkaMessageSensor(BaseSensorOperator):
    def __init__(self, topic, group_id, **kwargs):
        super().__init__(**kwargs)
        self.topic = topic
        self.group_id = group_id

    def poke(self, context):
        consumer = KafkaConsumerOperator(self.topic, self.group_id)
        msgs = consumer.consume_messages(max_messages=1)
        if msgs:
            context['ti'].xcom_push(key="kafka_message", value=msgs)
            return True
        return False

