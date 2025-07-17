# Use Python base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/
WORKDIR /app

# Expose port
EXPOSE 5050

# Run the API
CMD ["python", "app.py"]
