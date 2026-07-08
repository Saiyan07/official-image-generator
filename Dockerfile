FROM pytorch/pytorch:2.0.1-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# Copy all files from repo to container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (Hugging Face Spaces uses this port)
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
