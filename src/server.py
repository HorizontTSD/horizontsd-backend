from typing import Annotated, List

# import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.config import logger, public_or_local
from src.models.schemes import HellowRequest
from src.utils.greeting import hellow_names

if public_or_local == 'LOCAL':
    url = 'http://localhost'
else:
    url = 'http://11.11.11.11'

origins = [
    url
]

app = FastAPI(docs_url="/template_fast_api/v1/", openapi_url='/template_fast_api/v1/openapi.json')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""Функциональная часть отвечающая за отправку запроса на сайт и получение ответа в формате JSON"""
from fastapi import FastApi
import httpx

app = FastAPI()

@app.post("/")
async def graph_forecast (data: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://77.37.136.11:8501/", params=data)
        return response.json()