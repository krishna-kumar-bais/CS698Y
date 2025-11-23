##########################
# Stage 1: Build frontend #
##########################
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

##########################
# Stage 2: Python runtime #
##########################
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from stage 1 to Flask static dir
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Expose port
EXPOSE 5000

# Default command (Render sets $PORT). Use sh -c so $PORT expands
ENV PORT=5000
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT}"]
