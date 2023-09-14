from celery import Celery, signals
from model_loader import ModelLoader


def make_celery(app_name=__name__):
    backend = broker = 'redis://llama2_redis:6379/0'
    return Celery(app_name, backend=backend, broker=broker)

celery = make_celery()

model_loader = None

@signals.worker_process_init.connect
def setup_model(signal, sender, **kwargs):
    global model_loader
    model_loader = ModelLoader()

@celery.task
def generate_text_task(prompt):
    outputs = model_loader.llm(prompt=prompt)

    return outputs