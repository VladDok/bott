import http.client
import json
import time
import os

from dotenv import load_dotenv
load_dotenv()

API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")


def generate_text(prompt):
    conn = http.client.HTTPConnection(API_HOST, API_PORT)
    headers = {"Content-type": "application/json"}
    data = {"prompt": prompt}
    json_data = json.dumps(data)
    conn.request("POST", "/generate/", json_data, headers)
    response = conn.getresponse()
    result = json.loads(response.read().decode())
    conn.close()

    return result["task_id"]

def get_task_status(task_id):
    conn = http.client.HTTPConnection(API_HOST, API_PORT)
    conn.request("GET", f"/task/{task_id}")
    response = conn.getresponse()
    status = response.read().decode()
    conn.close()

    return status


if __name__ == "__main__":
    prompt = input("Enter the prompt: ")

    task_id = generate_text(prompt)
    while True:
        status = get_task_status(task_id)
        if "Task not completed yet" not in status:
            print(status)
            break
        time.sleep(2)