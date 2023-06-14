from fastapi import FastAPI
from llama_cpp import Llama
from typing import List, Optional
import os

MODEL_PATH = "./models/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin"
app = FastAPI()


@app.get("/", status_code=200)
async def handle_health_check():
    """ health check. This should be on root for default in vertex AI endpoint"""
    return {"health": "ok"}


@app.post("/predict")
async def handle_predict(query: str, echo: bool = False, stop: Optional[List[str]] = None):
    """ health check. This should be on root for default in vertex AI endpoint"""
    llm = Llama(model_path=MODEL_PATH)
    output = llm(query, max_tokens=200, echo=echo, stop=stop)
    print(output)
    return {"output": output["choices"][0]["text"]}