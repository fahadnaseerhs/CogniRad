#!/bin/bash
# Railway startup script that properly handles PORT environment variable

# Use PORT from environment, default to 8000 if not set
PORT=${PORT:-8000}

echo "Starting uvicorn on port $PORT"
exec uvicorn main:app --host 0.0.0.0 --port $PORT
