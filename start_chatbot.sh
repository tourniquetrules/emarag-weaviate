#!/bin/bash

# Emergency Medicine RAG Chatbot Startup Script
set -e

echo "🏥 Starting Emergency Medicine RAG Chatbot..."
echo "==========================================="

# Check if virtual environment exists
if [ ! -d "venv312" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected. Activating venv312..."
    source venv312/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if required files exist
if [ ! -f "medical_rag_chatbot.py" ]; then
    echo "❌ medical_rag_chatbot.py not found"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your settings"
fi

# Check if Weaviate is running
echo "🔍 Checking Weaviate status..."
if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo "✅ Weaviate is running"
else
    echo "❌ Weaviate is not running. Starting Docker Compose..."
    docker-compose up -d
    
    echo "⏳ Waiting for Weaviate to be ready..."
    timeout=60
    elapsed=0
    while ! curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; do
        if [ $elapsed -ge $timeout ]; then
            echo "❌ Timeout waiting for Weaviate to start"
            echo "💡 Try: docker-compose logs to check for errors"
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo ""
    echo "✅ Weaviate is ready"
fi

# Check GPU status (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1
fi

echo ""
echo "🚀 Launching Emergency Medicine RAG Chatbot..."
echo "🌐 Web interface will be available at: http://localhost:7871"
echo "⏹️  Press Ctrl+C to stop the chatbot"
echo ""

# Start the chatbot
python medical_rag_chatbot.py
