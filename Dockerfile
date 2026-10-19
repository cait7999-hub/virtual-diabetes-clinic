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

# Select the model version during Docker build
ARG MODEL_VERSION=v0.2
ENV MODEL_VERSION=${MODEL_VERSION}

# Train and save the selected model
RUN python train.py

# Expose API port
EXPOSE 80

# Start FastAPI application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
