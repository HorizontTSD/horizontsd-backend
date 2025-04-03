# Импортируем необходимые библиотеки.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# глобальные переменные приложения
app = FastAPI
EXT_URL  = "http://77.37.136.11:8501"

# Форма для заполнения клиентом(Model)
class UserData(BaseModel):
    """Ссюда необходимо занести данные которые будут подаваться со стороны клинта
    имена, числа, какие то другие данне необходимые для расчетов"""
    data: dict
    pass

@app.post("/")
async def my_request_rasponse(data: dict):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://77.37.136.11:8501/", params=data)
        return response.json()