#!/bin/bash

# Set script to exit on error
set -e

# Function to kill both processes when script exits
cleanup() {
    echo "Stopping all processes..."
    kill -TERM "$APP_PID" 2>/dev/null
    kill -TERM "$UVICORN_PID" 2>/dev/null
    wait "$APP_PID" "$UVICORN_PID" 2>/dev/null
    echo "Processes stopped."
}

# Trap Ctrl + C (SIGINT) and SIGTERM to run cleanup()
trap cleanup SIGINT SIGTERM

# Run the Python application in the background
echo "Running app.py..."
python3 app.py &
APP_PID=$!  # Capture the process ID of app.py

# Wait for a few seconds to ensure app.py initializes
sleep 3

# Run Uvicorn server in the background
echo "Starting Uvicorn server..."
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!  # Capture the process ID of uvicorn

# Wait for both processes
wait "$APP_PID" "$UVICORN_PID"
