FROM python:3.11

WORKDIR /app

COPY requirements.txt .  

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# First copy the entire project (excluding historical_data.csv if missing)
COPY . .

# Then explicitly copy historical_data.csv if it exists
RUN test -f historical_data.csv && cp historical_data.csv /app/historical_data.csv || echo "historical_data.csv not found, skipping..."

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
