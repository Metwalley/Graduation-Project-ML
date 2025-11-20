FROM python:3.9-slim

WORKDIR /app

# نسخ ملف المتطلبات وتحميلها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات (الكود والموديلات)
COPY . .

# فتح البورت
EXPOSE 8000

# تشغيل السيرفر
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]