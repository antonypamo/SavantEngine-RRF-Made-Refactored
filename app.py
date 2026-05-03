from fastapi import FastAPI
import requests

app = FastAPI()

BACKEND_URL = "https://antonypamo-apirrf.hf.space/evaluate"

@app.get("/")
def root():
    return {"status": "proxy_alive"}

@app.post("/")
async def evaluate(req: dict):
    try:
        response = requests.post(
            BACKEND_URL,
            json=req,
            timeout=25
        )

        return response.json()

    except Exception as e:
        return {
            "error": str(e),
            "type": "proxy_error"
        }
