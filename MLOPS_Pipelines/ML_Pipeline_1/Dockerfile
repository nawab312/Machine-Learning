FROM python:3.9-slim

WORKDIR /app

# Install gcc and other necessary dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash", "-c", "python src/preprocess.py && python src/train.py && python src/evaluate.py"]

