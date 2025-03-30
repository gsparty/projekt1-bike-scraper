FROM python:3.11

WORKDIR /app

COPY requirements.txt .  

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


COPY historical_data.csv /app/historical_data.csv 2>/dev/null || true

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
