from fastapi import FastAPI, Request
from llama_cpp import Llama
import os

AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')
MODEL_PATH = "./models/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin"

app = FastAPI()


# check requirements for custom container at: "https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements"
@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def handle_health_check():
    """ health check. This should be on root for default in vertex AI endpoint"""
    return {"health": "ok"}


@app.post(AIP_PREDICT_ROUTE)
async def handle_predict(request: Request):
    """ health check. This should be on root for default in vertex AI endpoint"""
    body = await request.json()
    outputs = []
    for question in body['instances']:
        llm = Llama(model_path=MODEL_PATH)
        output = llm(question['query'], max_tokens=200)
        outputs.append({"answer": output["choices"][0]["text"]})
    return {"predictions": outputs}
