# Build context for this Dockerfile is the PROJECT ROOT (not backend/),
# because backend/app.py imports sibling top-level modules like
# autocorrelation.py, cross_correlation.py, vessel_quantification.py, etc.
# docker-compose.yml is set up with that context already — if you ever run
# this manually, do it from the project root:
#   docker build -f backend/Dockerfile -t vessel-backend .
FROM python:3.11-slim

# System libraries needed at runtime by opencv-python-headless, scipy, and
# scikit-image (without these you'll hit "libGL.so.1: cannot open shared
# object file" or similar import errors the first time the backend runs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["python", "backend/app.py"]
