# Gunakan image Python terbaru
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements.txt ke dalam image
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode aplikasi ke dalam image
COPY . .

# Expose port yang digunakan aplikasi
EXPOSE 5000

# Jalankan aplikasi
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
