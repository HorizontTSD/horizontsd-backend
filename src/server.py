# src/server.py
import pandas as pd
from typing import Annotated, List, Dict
from fastapi import FastAPI, Body, HTTPException
from pydantic import ValidationError
from statsmodels.tsa.stattools import acf, pacf
from src.models.schemes import(
    ForecastResponse, 
    SensorData, 
    MapData, 
    MetricsTable,
    ForecastData,
    TimeSeriesInput
    )
from src.utils.data_processing import load_data, preprocess_data, train_model, predict, calculate_metrics, decompose_time_series, calculate_statistics

app = FastAPI(docs_url="/v1/docs", openapi_url='/v1/openapi.json')

@app.post("/v1/get_forecast_data", response_model=ForecastResponse)
async def get_forecast_data(
    body: Annotated[
        List[SensorData],
        Body(
            example=[
                {"time": "2024-09-06 12:00:00", "value": 123.45, "type": "True"},
                {"time": "2024-09-06 12:05:00", "value": 67.89, "type": "Input"}
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

        # Статистический анализ
        stats = calculate_statistics(df, target_column)

        # Разложение временного ряда
        decomposition = decompose_time_series(df, target_column, period=12)

        # Обучение модели
        model = train_model(df, target_column)

        # Прогнозирование
        forecast_df = df.copy()
        forecast_df["predicted_value"] = predict(model, forecast_df.drop(columns=[target_column]))

        # Вычисление метрик
        metrics = calculate_metrics(df[target_column], forecast_df["predicted_value"])

        # Формирование метрик для таблицы метрик
        metrics_tables = {
            "XGBoost": MetricsTable(
                text={"en": "XGBoost Metrics", "ru": "Метрики XGBoost"},
                values={"mse": float(metrics["mse"])}
            ),
            "LSTM": MetricsTable(
                text={"en": "LSTM Metrics", "ru": "Метрики LSTM"},
                values={"mse": float(metrics["mse"])}
            )
        }

        # Формирование таблицы для скачивания
        table_to_download = [
            SensorData(
                time=row.name.strftime("%Y-%m-%d %H:%M:%S"),
                value=float(row["value"]),
                type=str(row["type"])  # Преобразуем type в строку
            )
            for _, row in forecast_df.iterrows()
        ]

        # Формирование MapData
        map_data = MapData(
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
                }
            },
            table_to_download=table_to_download,
            metrix_tables=metrics_tables
        )

        # Формирование объекта ForecastData
        forecast_data = ForecastData(
            sensor_name="Датчик 1",
            sensor_id="id_1",
            map_data=map_data,
            table_to_download=table_to_download,
            metrix_tables=metrics_tables
        )

        # Возвращаем ответ в формате ForecastResponse
        return ForecastResponse(root={"sensor_1": forecast_data})

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Эндпоинт для основного графика временного ряда
@app.post("/v1/time_series/line_plot")
async def get_line_plot(
    body: Annotated[
        TimeSeriesInput,
        Body(
            example={
                "data": [
                    {"time": "2024-09-06 12:00:00", "load_consumption": 123.45},
                    {"time": "2024-09-06 12:05:00", "load_consumption": 67.89}
                ],
                "time_column": "time",
                "target_column": "load_consumption"
            }
        )
    ]
):
    df = pd.DataFrame(body.data)
    df[body.time_column] = pd.to_datetime(df[body.time_column])
    df.set_index(body.time_column, inplace=True)

    # Возвращаем данные для графика
    return {
        "x": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "y": df[body.target_column].tolist()
    }

# Эндпоинт для гистограмм по месяцам
@app.post("/v1/time_series/monthly_histogram")
async def get_monthly_histogram(
    body: Annotated[
        TimeSeriesInput,
        Body(
            example={
                "data": [
                    {"time": "2024-09-06 12:00:00", "load_consumption": 123.45},
                    {"time": "2024-09-06 12:05:00", "load_consumption": 67.89}
                ],
                "time_column": "time",
                "target_column": "load_consumption"
            }
        )
    ]
):
    df = pd.DataFrame(body.data)
    df[body.time_column] = pd.to_datetime(df[body.time_column])
    df.set_index(body.time_column, inplace=True)

    # Группировка по месяцам и вычисление максимума
    monthly_data = df.resample('M').agg({body.target_column: 'max'}).reset_index()

    return {
        "months": monthly_data[body.time_column].dt.strftime('%B').tolist(),
        "values": monthly_data[body.target_column].tolist()
    }

# эндпоинт для таблицы статистических показателей
@app.post("/v1/time_series/monthly_statistics")
async def get_monthly_statistics(
    body: Annotated[
        TimeSeriesInput,
        Body(
            example={
                "data": [
                    {"time": "2024-09-06 12:00:00", "load_consumption": 123.45},
                    {"time": "2024-09-06 12:05:00", "load_consumption": 67.89}
                ],
                "time_column": "time",
                "target_column": "load_consumption"
            }
        )
    ]
):
    df = pd.DataFrame(body.data)
    df[body.time_column] = pd.to_datetime(df[body.time_column])
    df.set_index(body.time_column, inplace=True)

    # Группировка по месяцам и вычисление статистики
    monthly_stats = df.resample('M').agg({
        body.target_column: ['max', 'min', 'mean', 'median', 'std']
    }).reset_index()

    # Переименование столбцов
    monthly_stats.columns = ['month', 'max', 'min', 'mean', 'median', 'std_dev']

    return {
        "months": monthly_stats['month'].dt.strftime('%B').tolist(),
        "stats": monthly_stats[['max', 'min', 'mean', 'median', 'std_dev']].to_dict(orient='records')
    }

# Эндпоинт для Boxplot по месяцам
@app.post("/v1/time_series/monthly_boxplot")
async def get_monthly_boxplot(
    body: Annotated[
        TimeSeriesInput,
        Body(
            example={
                "data": [
                    {"time": "2024-09-06 12:00:00", "load_consumption": 123.45},
                    {"time": "2024-09-06 12:05:00", "load_consumption": 67.89}
                ],
                "time_column": "time",
                "target_column": "load_consumption"
            }
        )
    ]
):
    df = pd.DataFrame(body.data)
    df[body.time_column] = pd.to_datetime(df[body.time_column])
    df.set_index(body.time_column, inplace=True)

    # Добавляем колонку с месяцами
    df['month'] = df.index.strftime('%B')

    # Группировка по месяцам
    monthly_data = df.groupby('month')[body.target_column].apply(list).to_dict()

    return {
        "months": list(monthly_data.keys()),
        "data": list(monthly_data.values())
    }

#  Эндпоинт для ACF/PACF диаграмм
@app.post("/v1/time_series/acf_pacf")
async def get_acf_pacf(
    body: Annotated[
        TimeSeriesInput,
        Body(
            example={
                "data": [
                    {"time": "2024-09-06 12:00:00", "load_consumption": 123.45},
                    {"time": "2024-09-06 12:05:00", "load_consumption": 67.89}
                ],
                "time_column": "time",
                "target_column": "load_consumption"
            }
        )
    ]
):
    df = pd.DataFrame(body.data)
    df[body.time_column] = pd.to_datetime(df[body.time_column])
    df.set_index(body.time_column, inplace=True)

    # Проверка размера данных
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data points for ACF/PACF calculation.")

    # Определяем максимальное значение для nlags
    max_lags = len(df) // 2 - 1
    lags = min(30, max_lags)  # Берем минимальное значение между 30 и max_lags

    if lags <= 0:
        raise HTTPException(status_code=400, detail="Insufficient data points for ACF/PACF calculation.")

    # Вычисляем ACF и PACF
    acf_values = acf(df[body.target_column], nlags=lags)
    pacf_values = pacf(df[body.target_column], nlags=lags)

    return {
        "lags": list(range(lags + 1)),
        "acf": acf_values.tolist(),
        "pacf": pacf_values.tolist()
    }


@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}