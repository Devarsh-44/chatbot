# Use a base image with Python 3.12
FROM python:3.12-slim

# Install system dependencies to ensure distutils is available
RUN apt-get update && apt-get install -y \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Define the command to run your app
# Replace "app.py" with the entry point of your application (e.g., frontend.py or backend.py)
CMD ["python", "app.py"]