# Импортируем необходимые библиотеки.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# глобальные переменные приложения
app = FastAPI
GORIZONT_URL  = "http://77.37.136.11:8501"

# Форма для заполнения клиентом(Model)
class UserData(BaseModel):
    """Ссюда необходимо занести данные которые будут подаваться со стороны клинта
    имена, числа, какие то другие данне необходимые для расчетов"""
    data: dict
    pass

@app.post("/process")
async def process_data(request: ClientRequest):
    """В данной функции(обработчик маршрута) принимаю запрос данными,
        так же нужно отправить на внешний API, делает запрос и 
        возращаем ответ клиенту"""
   async with httpx.AsyncClient() as client:
        try:
            post_response = await client.post(
                GORIZONT_URL,
                json={"data": request.data}
            )
            post_response.raise_for_status()
            get_response = await client.get(GORIZONT_URL)
            get_response.raise_for_status()

            return get_response.json()