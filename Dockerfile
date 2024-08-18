# Gunakan image Python sebagai base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Salin requirements.txt dan instal dependensi
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Salin seluruh kode aplikasi
COPY . .

# Jalankan aplikasi Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
