# Импортируем необходимые библиотеки.
from pydantic import BaseModel
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.config import logger, public_or_local
from src.models.schemes import HellowRequest
from src.utils.greeting import hellow_names

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
        
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Внешняя ошибка API: {str(e)}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка соединения: {str(e)}"
            )
@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}
       

if __name__ == "__main__":
    import uvicorn
    port = 7070
    uvicorn.run(app, host="0.0.0.0", port= port)