from typing import Annotated, List, Dict, Optional
from datetime import date
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

from src.config import logger, public_or_local

# Конфигурация
url = 'http://localhost' if public_or_local == 'LOCAL' else 'http://11.11.11.11'

app = FastAPI(
    docs_url="/template_fast_api/v1/",
    openapi_url='/template_fast_api/v1/openapi.json',
    title="Data Analysis API",
    description="API for data processing and visualization"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Схемы данных
class TimeSeriesRequest(BaseModel):
    start_date: date
    end_date: date
    metrics: List[str]

class ChartDataResponse(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, List[float]]]

class StatisticalResponse(BaseModel):
    mean: float
    median: float
    std_dev: float
    min: float
    max: float

class AnalysisRequest(BaseModel):
    data: List[float]
    bins: Optional[int] = 10

SAMPLE_DATA = {
    'dates': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
    'values': np.random.randn(365).cumsum()
}

@app.post("/template_fast_api/v1/analyze/statistics")
async def calculate_statistics(request: AnalysisRequest):
    """Возвращает статистические показаний данных но поэтому много вопросов"""
    try:
        data = np.array(request.data)
        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std_dev": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data))
        }
    except Exception as e:
        logger.error(f"Statistics calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating statistics")

@app.post("/template_fast_api/v1/analyze/histogram")
async def generate_histogram(request: AnalysisRequest):
    """Генерирует данные для гистограммы"""
    try:
        hist, bins = np.histogram(request.data, bins=request.bins)
        return {
            "frequencies": hist.tolist(),
            "bin_edges": bins.tolist()
        }
    except Exception as e:
        logger.error(f"Histogram generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating histogram data")

@app.get("/template_fast_api/v1/data/timeseries")
async def get_time_series_data():
    """Возвращает временной ряд с примерами данных"""
    try:
        df = pd.DataFrame(SAMPLE_DATA)
        return {
            "labels": df['dates'].dt.strftime('%Y-%m-%d').tolist(),
            "datasets": [{
                "label": "Sample Time Series",
                "data": df['values'].tolist()
            }]
        }
    except Exception as e:
        logger.error(f"Time series error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching time series data")

@app.post("/template_fast_api/v1/analyze/correlation")
async def calculate_correlation(request: TimeSeriesRequest):
    """Рассчитывает корреляцию между метриками"""
    try:
        # Здесь должна быть ваша реальная логика получения данных
        dummy_data = {
            'metric1': np.random.rand(100),
            'metric2': np.random.rand(100) + 0.5
        }
        df = pd.DataFrame(dummy_data)
        correlation = df.corr().values.tolist()
        return {
            "correlation_matrix": correlation,
            "labels": request.metrics
        }
    except Exception as e:
        logger.error(f"Correlation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating correlation")

@app.get("/")
def read_root():
    return {"message": "Welcome to Data Analysis API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7070)