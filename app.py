from fastapi import FastAPI
from celery.result import AsyncResult
from celery_worker import generate_text_task


app = FastAPI()

@app.post("/generate/")
async def generate_text(item):
    task = generate_text_task.delay(item.prompt)
    return {"task_id": task.id}

@app.get("/task/{task_id}")
async def get_task(task_id):
    result = AsyncResult(task_id)
    if result.ready():
        response = result.get()
        return {"result": response}
    else:
        return {"status": "Task not completed yet"}