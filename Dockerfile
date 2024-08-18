# Gunakan image dasar Python
FROM python:3.11-slim

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt dan install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
