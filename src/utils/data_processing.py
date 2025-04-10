# src/utils/data_processing.py
import pandas as pd
import numpy as np
import httpx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

def load_data(data: list) -> pd.DataFrame:
    """
    Преобразует входные данные в DataFrame.
    :param data: Список объектов SensorData.
    :return: DataFrame с временными метками в качестве индекса.
    """
    # Преобразуем объекты Pydantic в словари
    data_as_dicts = [item.dict() for item in data]

    # Проверяем наличие поля 'time'
    if not data_as_dicts or 'time' not in data_as_dicts[0]:
        raise ValueError("Field 'time' is missing in the input data.")

    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data_as_dicts)

    # Преобразуем время в формат datetime
    try:
        df['time'] = pd.to_datetime(df['time'], errors='raise')
    except Exception as e:
        raise ValueError(f"Failed to parse 'time' field: {e}")

    # Устанавливаем время как индекс
    df.set_index('time', inplace=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка данных: заполнение пропусков, нормализация и т.д.
    :param df: Исходный DataFrame.
    :return: Обработанный DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty after loading data.")

    # Проверяем наличие столбца 'value'
    if 'value' not in df.columns:
        raise ValueError("Column 'value' is missing in the DataFrame.")

    # Заполняем пропущенные значения
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Преобразуем столбец 'type' в числовой формат
    label_encoder = LabelEncoder()
    df['type'] = label_encoder.fit_transform(df['type'])

    # Добавляем новые признаки
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour

    return df

def train_model(df: pd.DataFrame, target_column: str) -> RandomForestRegressor:
    """
    Обучает модель машинного обучения на основе данных.
    :param df: DataFrame с признаками и целевой переменной.
    :param target_column: Название столбца с целевой переменной.
    :return: Обученная модель.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing in the DataFrame.")

    X = df.drop(columns=[target_column])  # Признаки
    y = df[target_column]                 # Целевая переменная

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оцениваем качество модели на тестовой выборке
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")

    return model

def predict(model, df: pd.DataFrame) -> pd.Series:
    """
    Выполняет прогнозирование значений на основе обученной модели.
    :param model: Обученная модель.
    :param df: DataFrame с признаками.
    :return: Series с прогнозируемыми значениями.
    """
    return pd.Series(model.predict(df), index=df.index)

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Вычисляет метрики качества прогноза.
    :param y_true: Реальные значения.
    :param y_pred: Прогнозируемые значения.
    :return: Словарь с метриками.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {"mse": mse}