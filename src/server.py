# src/server.py
import pandas as pd
from src.utils.data_processing import prepare_time_series_dataframe
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
    """
    Returns data for a time series line plot.
    This endpoint provides x and y values for visualizing a time series.
    """
    # Преобразуем входные данные в DataFrame
    df = prepare_time_series_dataframe(body.data, body.time_column, body.target_column)

    # Формируем ответ с данными для графика и подписями
    return {
        "legend": {
            "title": {
                "en": "Time Series Line Plot",
                "ru": "График временного ряда"
            },
            "x_axis": {
                "en": "Time",
                "ru": "Время"
            },
            "y_axis": {
                "en": "Value",
                "ru": "Значение"
            }
        },
        "x": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),  # Временные метки
        "y": df[body.target_column].tolist()  # Значения целевой переменной
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
    """
    Returns monthly histogram data for visualization.
    This endpoint groups the input data by month and calculates the maximum value 
    for each month. If there is no data for a specific month, it returns `None` 
    for that month. The response includes localized legends and month numbers.
    """
    # Преобразуем входные данные в DataFrame
    df = prepare_time_series_dataframe(body.data, body.time_column, body.target_column)
    # Группировка по месяцам и вычисление максимума
    monthly_data = df.resample('M').agg({body.target_column: 'max'}).reset_index()
    # Список всех месяцев
    all_months = pd.date_range(start="2024-01-01", end="2024-12-31", freq='MS')
    monthly_data['month'] = monthly_data[body.time_column].dt.strftime('%B')
    monthly_data['month_number'] = monthly_data[body.time_column].dt.strftime('%m')
    # Формируем ответ с учетом всех месяцев
    response_months = []
    response_values = []
    response_month_numbers = []
    for month_date in all_months:
        month_name = month_date.strftime('%B')
        month_number = month_date.strftime('%m')
        if month_name in monthly_data['month'].values:
            value = monthly_data.loc[monthly_data['month'] == month_name, body.target_column].values[0]
        else:
            value = None
        response_months.append(month_name)
        response_values.append(value)
        response_month_numbers.append(month_number)
    return {
        "legend": {
            "title": {
                "en": "Monthly Load Consumption",
                "ru": "Потребление нагрузки по месяцам"
            },
            "x_axis": {
                "en": "Months",
                "ru": "Месяцы"
            },
            "y_axis": {
                "en": "Load Consumption",
                "ru": "Потребление нагрузки"
            }
        },
        "months": response_months,
        "month_numbers": response_month_numbers,  # Добавляем номера месяцев
        "values": response_values
    }

# Эндпоинт для таблицы статистических показателей
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
    """
    Returns monthly statistics for visualization.
    This endpoint groups the input data by month and calculates statistical metrics 
    (max, min, mean, median, std) for each month. If there is no data for a specific month, 
    it returns `None` for that month. The response includes localized legends and month numbers.
    """
    # Преобразуем входные данные в DataFrame
    df = prepare_time_series_dataframe(body.data, body.time_column, body.target_column)
    
    # Группировка по месяцам и вычисление статистики
    monthly_stats = df.resample('M').agg({
        body.target_column: ['max', 'min', 'mean', 'median', 'std']
    }).reset_index()
    
    # Переименование столбцов
    monthly_stats.columns = ['month', 'max', 'min', 'mean', 'median', 'std_dev']
    
    # Список всех месяцев
    all_months = pd.date_range(start="2024-01-01", end="2024-12-31", freq='MS')
    monthly_stats['month'] = monthly_stats['month'].dt.strftime('%B')  # Название месяца
    monthly_stats['month_number'] = monthly_stats['month'].apply(lambda x: pd.to_datetime(x, format='%B').strftime('%m'))  # Номер месяца
    
    # Формируем ответ с учетом всех месяцев
    response_months = []
    response_month_numbers = []
    response_stats = []
    for month_date in all_months:
        month_name = month_date.strftime('%B')
        month_number = month_date.strftime('%m')
        if month_name in monthly_stats['month'].values:
            stats = monthly_stats.loc[monthly_stats['month'] == month_name].iloc[0]
            response_stats.append({
                "max": stats["max"],
                "min": stats["min"],
                "mean": stats["mean"],
                "median": stats["median"],
                "std_dev": stats["std_dev"]
            })
        else:
            response_stats.append({
                "max": None,
                "min": None,
                "mean": None,
                "median": None,
                "std_dev": None
            })
        response_months.append(month_name)
        response_month_numbers.append(month_number)
    
    return {
        "legend": {
            "title": {
                "en": "Monthly Statistics",
                "ru": "Статистика по месяцам"
            },
            "x_axis": {
                "en": "Months",
                "ru": "Месяцы"
            },
            "y_axis": {
                "en": "Statistical Values",
                "ru": "Статистические значения"
            }
        },
        "months": response_months,  # Названия месяцев
        "month_numbers": response_month_numbers,  # Номера месяцев
        "stats": response_stats  # Статистические показатели
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
    """
    Returns data for a monthly boxplot.
    This endpoint groups the input data by month and provides the values for each month.
    If there is no data for a specific month, it returns an empty list for that month.
    The response includes localized legends and month numbers for easier mapping.
    """
    # Преобразуем входные данные в DataFrame
    df = prepare_time_series_dataframe(body.data, body.time_column, body.target_column)

    # Добавляем колонку с месяцами
    df['month'] = df.index.strftime('%B')
    df['month_number'] = df.index.strftime('%m')  # Добавляем номер месяца

    # Группировка по месяцам
    monthly_data = df.groupby('month')[body.target_column].apply(list).to_dict()
    monthly_numbers = df.groupby('month')['month_number'].first().to_dict()  # Номера месяцев

    # Список всех месяцев
    all_months = pd.date_range(start="2024-01-01", end="2024-12-31", freq='MS')
    all_month_names = all_months.strftime('%B').tolist()
    all_month_numbers = all_months.strftime('%m').tolist()

    # Формируем ответ с учетом всех месяцев
    response_months = []
    response_month_numbers = []
    response_data = []
    for month_name, month_number in zip(all_month_names, all_month_numbers):
        if month_name in monthly_data:
            response_data.append(monthly_data[month_name])
        else:
            response_data.append([])  # Пустой список, если данных за месяц нет
        response_months.append(month_name)
        response_month_numbers.append(month_number)

    return {
        "legend": {
            "title": {
                "en": "Monthly Boxplot",
                "ru": "Боксплот по месяцам"
            },
            "x_axis": {
                "en": "Months",
                "ru": "Месяцы"
            },
            "y_axis": {
                "en": "Values",
                "ru": "Значения"
            }
        },
        "months": response_months,          # Названия месяцев
        "month_numbers": response_month_numbers,  # Номера месяцев
        "data": response_data               # Данные для боксплота
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
    """
    Returns ACF and PACF values for time series analysis.
    This endpoint calculates autocorrelation (ACF) and partial autocorrelation (PACF)
    for the given time series data. If there are insufficient data points, an error is raised.
    """
    # Преобразуем входные данные в DataFrame
    df = prepare_time_series_dataframe(body.data, body.time_column, body.target_column)

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
        "legend": {
            "title": {
                "en": "ACF and PACF Analysis",
                "ru": "Анализ ACF и PACF"
            },
            "x_axis": {
                "en": "Lags",
                "ru": "Лаги"
            },
            "y_axis": {
                "en": "Correlation",
                "ru": "Корреляция"
            }
        },
        "lags": list(range(lags + 1)),
        "acf": acf_values.tolist(),
        "pacf": pacf_values.tolist()
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}