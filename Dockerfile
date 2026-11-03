FROM python:3.10-slim

WORKDIR /app

# Instala ferramentas de compilação e o pacote locales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Configura o locale necessário para o ONNX Runtime processar texto
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8 
ENV LC_ALL en_US.UTF-8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY tests/ ./tests/

# Gera os artefatos dos modelos no momento do build
RUN python scripts/generate_data.py && python scripts/train_model.py && python scripts/optimize_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]