# src/server.py
from typing import Annotated, List, Dict
from fastapi import FastAPI, Body, HTTPException
from pydantic import ValidationError
from src.models.schemes import ForecastResponse, SensorData, MapData, MetricsTable  # Импортируем все необходимые модели
from src.utils.data_processing import load_data, preprocess_data, train_model, predict, calculate_metrics

app = FastAPI(docs_url="/v1/docs", openapi_url='/v1/openapi.json')

@app.post("/v1/get_forecast_data", response_model=ForecastResponse)
async def get_forecast_data(
    body: Annotated[
        List[SensorData],
        Body(
            example=[
                {"time": "2024-09-06 12:00:00", "value": 123.45, "type": "True"},
                {"time": "2024-09-06 12:05:00", "value": 67.89, "type": "Input"},
                {"time": "2024-09-06 12:10:00", "value": 101.12, "type": "True"}
            ]
        )
    ]
):
    try:
        # Загрузка и предобработка данных
        df = load_data(body)
        df = preprocess_data(df)

        # Определение целевой переменной
        target_column = "value"

        # Обучение модели
        model = train_model(df, target_column)

        # Прогнозирование
        forecast_df = df.copy()
        forecast_df["predicted_value"] = predict(model, forecast_df.drop(columns=[target_column]))

        # Вычисление метрик
        metrics = calculate_metrics(df[target_column], forecast_df["predicted_value"])

        # Формирование данных для ответа
        forecast_data = {
            "sensor_name": "Датчик 1",
            "sensor_id": "id_1",
            "map_data": MapData(
                data={
                    "last_real_data": float(df.iloc[-1][target_column]),
                    "actual_prediction_lstm": float(forecast_df.iloc[-1]["predicted_value"]),
                    "actual_prediction_xgboost": float(forecast_df.iloc[-1]["predicted_value"]),
                    "ensemble": float(forecast_df.iloc[-1]["predicted_value"])
                },
                last_know_data=df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                legend={
                    "last_know_data_line": {
                        "text": {"en": "Last Known Data", "ru": "Последние известные данные"},
                        "color": "#FF0000"
                    },
                    "real_data_line": {
                        "text": {"en": "Real Data", "ru": "Реальные данные"},
                        "color": "#00FF00"
                    },
                    "LSTM_data_line": {
                        "text": {"en": "LSTM Prediction", "ru": "Прогноз LSTM"},
                        "color": "#0000FF"
                    },
                    "XGBoost_data_line": {
                        "text": {"en": "XGBoost Prediction", "ru": "Прогноз XGBoost"},
                        "color": "#FFFF00"
                    },
                    "Ensemble_data_line": {
                        "text": {"en": "Ensemble Prediction", "ru": "Прогноз Ensemble"},
                        "color": "#00FFFF"
                    }
                },
                table_to_download=[],  
                metrix_tables={}       
            ),
            "table_to_download": [
                SensorData(
                    time=row.name.strftime("%Y-%m-%d %H:%M:%S"),
                    value=float(row["value"]),
                    type=str(int(row["type"]))
                )
                for _, row in forecast_df.iterrows()
            ],
            "metrix_tables": {
                "XGBoost": MetricsTable(
                    text={"en": "XGBoost Metrics", "ru": "Метрики XGBoost"},
                    values={"mse": float(metrics["mse"])}
                ),
                "LSTM": MetricsTable(
                    text={"en": "LSTM Metrics", "ru": "Метрики LSTM"},
                    values={"mse": float(metrics["mse"])}
                )
            }
        }


        # Создаем объект ForecastResponse
        response = ForecastResponse(root={"sensor_1": forecast_data})

        print("ForecastData:", forecast_data)
        print("ForecastResponse:", response)
        return response

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}