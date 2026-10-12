import os
import requests
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from datetime import timedelta
from langchain_core.tools import tool
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from chronos import BaseChronosPipeline

load_dotenv()

DB_PARAMS = {
        "dbname": "data_prices",
        "user": "postgres",
        "password": os.getenv("DB_PASSWORD"),
        "host": "localhost",
        "port": "5432"
    }

@tool
def html2text_tool(url: str) -> str:
    """Fetches a URL asynchronously and converts its HTML content into clean markdown text."""
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed[0].page_content if docs_transformed else "No content found."

@tool
def fetch_usd_rub_rate(currency: str = "RUB") -> float | str:
    """
    Получает текущий актуальный курс доллара США к заданной валюте.
    """
    url = "https://open.er-api.com/v6/latest/USD"
    target_currency = currency.upper()
    
    try:
        response = requests.get(url).json()
        if response.get("result") == "success":
            rates = response.get("rates", {})
            rate_value = rates.get(target_currency)
            if rate_value is not None:
                return round(float(rate_value), 2)
        return 0.0
    except Exception as e:
        return f"Ошибка при запросе курса валют: {str(e)}"

@tool 
def save_rate_to_db(rate_date: str, rate: float, currency_pair: str = "USD/RUB") -> str:
    """
    Сохраняет или обновляет курс USD к заданной валюте в базе данных PostgreSQL за конкретную дату.
    Формат даты: YYYY-MM-DD.
    """

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS usd_rates (
                id SERIAL PRIMARY KEY,
                rate_date DATE NOT NULL UNIQUE,
                rate NUMERIC(10, 4) NOT NULL,
                currency_pair VARCHAR(10) DEFAULT '{currency_pair}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute(f"""
            INSERT INTO usd_rates (rate_date, rate, currency_pair) 
            VALUES (%s, %s, %s)
            ON CONFLICT (rate_date) DO UPDATE SET rate = EXCLUDED.rate;
        """, (rate_date, rate, currency_pair))
        
        conn.commit()
        cursor.close()
        conn.close()
        return f"Успешно записано в БД: Дата {rate_date}, Курс {rate} {currency_pair}."
    except Exception as e:
        return f"Ошибка работы с базой данных: {str(e)}"

@tool
def forecast_usd_rub_next_week(currency_pair: str = "USD/RUB") -> str:
    """
    Выгружает историю курсов валют из PostgreSQL, скачивает модель 
    amazon/chronos-2 с Hugging Face и прогнозирует курс на 7 дней вперед.
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        query = f"""
            SELECT rate_date, rate 
            FROM usd_rates 
            WHERE currency_pair = '{currency_pair}'
            ORDER BY rate_date ASC;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
           
        context_df = pd.DataFrame()
        context_df["item_id"] = [currency_pair] * len(df)
        context_df["rate_date"] = pd.to_datetime(df["rate_date"])
        context_df["rate"] = df["rate"].astype(float)
        
        device = "cuda" if os.getenv("HAS_CUDA") else "cpu" 
        
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", 
            device_map=device
        )
        
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=7,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="rate_date",
            target="rate",
        )

        last_date = context_df["rate_date"].max()
        report = f"Прогноз курса {currency_pair} на следующие 7 дней:\n"
        
        for idx, row in pred_df.iterrows():
            future_date = (last_date + timedelta(days=idx + 1)).strftime('%Y-%m-%d')
            
            p_low = round(float(row["0.1"]), 2)
            p_med = round(float(row["0.5"]), 2)
            p_high = round(float(row["0.9"]), 2)
            
            report += f"{future_date}: {p_med} {currency_pair}. (диапазон: {p_low} - {p_high})\n"
            
        return report

    except Exception as e:
        return f"Ошибка в процессе прогнозирования: {str(e)}"
