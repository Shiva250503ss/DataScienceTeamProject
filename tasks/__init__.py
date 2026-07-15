# tasks/ - Celery async task layer.
# Worker entry point (matches docker-compose.prod.yml):
#   celery -A tasks.worker worker --loglevel=info            (Linux/containers)
#   celery -A tasks.worker worker --pool=solo --loglevel=info (Windows dev)

from tasks.worker import celery_app, run_pipeline_task  # noqa: F401
