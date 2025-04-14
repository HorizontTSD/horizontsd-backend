# src/utils/data_processing.py
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def prepare_time_series_dataframe(
    data: List[Dict],
    time_column: str,
    target_column: str
) -> pd.DataFrame:
    """
    Converts a list of time series records into a pandas DataFrame
    and sets the correct time index.

    Args:
        data (List[Dict]): Raw input data.
        time_column (str): Name of the time column.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with datetime index.
    """
    df = pd.DataFrame(data)
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    return df[[target_column]]

def calculate_statistics(df: pd.DataFrame, target_column: str) -> dict:
    """
    Вычисляет описательную статистику для временного ряда.
    :param df: DataFrame с данными.
    :param target_column: Название целевого столбца.
    :return: Словарь с описательной статистикой.
    """
    stats = {
        "min": float(df[target_column].min()),
        "max": float(df[target_column].max()),
        "mean": float(df[target_column].mean()),
        "median": float(df[target_column].median()),
        "std": float(df[target_column].std())
    }
    return stats

def decompose_time_series(df: pd.DataFrame, target_column: str, period: int = 24):
    """
    Разлагает временной ряд на тренд, сезонность и остатки.
    :param df: DataFrame с данными.
    :param target_column: Название целевого столбца.
    :param period: Период сезонности (например, 24 часа).
    :return: Результат декомпозиции.
    """
    if len(df) < 2 * period:
        return {
            "trend": {},
            "seasonal": {},
            "residual": {}
        }
    
    result = seasonal_decompose(df[target_column], model='additive', period=period)
    return {
        "trend": result.trend.dropna().to_dict(),
        "seasonal": result.seasonal.dropna().to_dict(),
        "residual": result.resid.dropna().to_dict()
    }

def plot_autocorrelation(df: pd.DataFrame, target_column: str, lags: int = 50):
    """
    Строит графики ACF и PACF.
    :param df: DataFrame с данными.
    :param target_column: Название целевого столбца.
    :param lags: Количество лагов.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_acf(df[target_column], lags=lags, ax=axes[0])
    plot_pacf(df[target_column], lags=lags, ax=axes[1])
    plt.tight_layout()
    plt.show()

def visualize_time_series(df: pd.DataFrame, target_column: str):
    """
    Визуализирует временной ряд.
    :param df: DataFrame с данными.
    :param target_column: Название целевого столбца.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=df.index, y=target_column)
    plt.title("Time Series Visualization")
    plt.xlabel("Time")
    plt.ylabel(target_column)
    plt.show()

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