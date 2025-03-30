FROM python:3.11

WORKDIR /app

COPY requirements.txt .  

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire project EXCEPT `historical_data.csv` (to prevent errors)
COPY . .

# Ensure the CSV file is copied only if it exists
ARG CSV_FILE=historical_data.csv
RUN if [ -f "$CSV_FILE" ]; then cp "$CSV_FILE" /app/; else echo "historical_data.csv not found, skipping..."; fi

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
