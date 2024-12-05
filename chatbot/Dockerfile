ARG PYTHON_VERSION=3.11-slim

FROM python:${PYTHON_VERSION}

WORKDIR /ai_fastapi_app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .

CMD ["python3","main.py"]