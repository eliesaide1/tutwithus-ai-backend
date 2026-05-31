FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    # xmlsec / lxml
    libxml2-dev \
    libxslt-dev \
    libxmlsec1-dev \
    libxmlsec1-openssl \
    pkg-config \
    # OpenCV
    libgl1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    # pytesseract / passporteye
    tesseract-ocr \
    # psycopg C extension
    libpq-dev \
    # build tools
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --upgrade pip && pip install -e . --target /install

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    # xmlsec / lxml
    libxml2-dev \
    libxslt-dev \
    libxmlsec1-dev \
    libxmlsec1-openssl \
    pkg-config \
    # OpenCV
    libgl1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    # pytesseract / passporteye
    tesseract-ocr \
    # psycopg C extension
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY . /app
WORKDIR /app

EXPOSE 8001

CMD uvicorn api.Apexchat_api:app --host 0.0.0.0 --port ${PORT:-8001}
