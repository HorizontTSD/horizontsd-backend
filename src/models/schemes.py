# src/models/schemes.py
from typing import Dict, List
from pydantic import BaseModel, RootModel


class LegendItem(BaseModel):
    """
    Модель для элемента легенды графика.
    """
    text: Dict[str, str]  # Описание на русском и английском языках
    color: str  # Цвет линии (например, "#FF0000")

class TimeSeriesInput(BaseModel):
    """
    Модель для входных данных временного ряда.
    """
    data: List[Dict]  # Список словарей с данными
    time_column: str  # Название колонки времени
    target_column: str  # Название целевой колонки

class SensorData(BaseModel):
    """
    Модель данных датчика.
    """
    time: str  # Временная метка (например, "2024-09-06 12:00:00")
    value: float  # Значение датчика
    type: str  # Тип данных (например, "True" или "Input")

class MapData(BaseModel):
    """
    Модель для данных карты/графиков.
    """
    data: Dict[str, float]  # Данные для отрисовки графиков
    last_know_data: str  # Последняя известная дата
    legend: Dict[str, LegendItem]  # Легенда графика
    table_to_download: List[SensorData]  # Таблица данных для скачивания
    metrix_tables: Dict[str, "MetricsTable"]  # Метрики моделей

class MetricsTable(BaseModel):
    """
    Модель для метрик моделей.
    """
    text: Dict[str, str]  # Описание на русском и английском языках
    values: Dict[str, float]  # Значения метрик (например, MSE)

class ForecastData(BaseModel):
    """
    Модель для данных прогноза одного датчика.
    """
    sensor_name: str  # Отображаемое имя датчика
    sensor_id: str  # ID датчика
    map_data: MapData  # Данные для визуализации
    table_to_download: List[SensorData]  # Таблица данных для скачивания
    metrix_tables: Dict[str, MetricsTable]  # Метрики моделей

class ForecastResponse(RootModel):
    """
    Модель для ответа API.
    """
    root: Dict[str, ForecastData]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]