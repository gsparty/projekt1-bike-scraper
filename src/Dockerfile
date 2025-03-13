FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Ensure the historical data file is copied
COPY historical_data.csv /app/historical_data.csv

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]