# Gunakan image dasar yang memiliki Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Salin requirements.txt dan install dependencies
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam container


# Expose port aplikasi
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "app.py"]
