FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY analysis.py .
COPY preprocess.py .
COPY tests ./tests
COPY .flake8 .

CMD ["python", "-m", "pytest", "-v"]