# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system build dependencies for NumPy and SciPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and core tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose both FastAPI and Streamlit ports
EXPOSE 8000 8501

# Run both backend (FastAPI) and frontend (Streamlit) in one container
CMD uvicorn services.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
