# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train model inside Docker image
RUN python train.py

# Expose API port
EXPOSE 8000

# Start FastAPI application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
