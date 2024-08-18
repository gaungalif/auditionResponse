# Gunakan base image yang mendukung Python 3.9
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Tentukan working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependencies Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .

# Jalankan perintah untuk Celery worker
CMD ["celery", "-A", "app", "worker", "--loglevel=info"]
