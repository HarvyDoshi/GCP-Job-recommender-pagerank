# Use official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader stopwords && \
    python -m nltk.downloader punkt

# Copy application files
COPY . .

# Expose the port explicitly
EXPOSE 8501

# Command to run the application (no env variable in CMD)
CMD ["streamlit", "run", "str.py", "--server.port=8501", "--server.address=0.0.0.0"]
