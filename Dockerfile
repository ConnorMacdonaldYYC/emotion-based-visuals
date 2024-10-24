# syntax=docker/dockerfile:1

# Use Python 3.11 slim image as the base
FROM python:3.11-slim

RUN apt-get update && apt-get -y install build-essential cmake libopencv-dev

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY image_generation/ latentblending/ server.py .

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the app
CMD ["python", "server.py"]