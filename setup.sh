#!/bin/bash

# Emergency Medicine RAG Chatbot Setup Script
# This script sets up the complete environment for the emarag-weaviate project

set -e  # Exit on any error

echo "🏥 Emergency Medicine RAG Chatbot Setup"
echo "======================================"

# Check if Python 3.12+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.12"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    echo "❌ Python 3.12+ is required. Current version: $python_version"
    echo "Please install Python 3.12 or later and try again."
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker and try again."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker check passed"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker Compose check passed"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv312" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv312
fi

python3 -m venv venv312
source venv312/bin/activate

echo "✅ Virtual environment created"

# Upgrade pip
echo ""
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Python dependencies installed"

# Download spaCy model
echo ""
echo "🧠 Downloading spaCy medical model..."
python -m spacy download en_core_sci_scibert

echo "✅ spaCy model downloaded"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "📝 Please edit .env file with your API keys and settings"
else
    echo "✅ .env file already exists"
fi

# Start Weaviate database
echo ""
echo "🗄️  Starting Weaviate database..."
docker-compose up -d

# Wait for Weaviate to be ready
echo "⏳ Waiting for Weaviate to be ready..."
timeout=60
elapsed=0
while ! curl -f http://localhost:8080/v1/meta > /dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        echo "❌ Timeout waiting for Weaviate to start"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo -n "."
done

echo ""
echo "✅ Weaviate database is running"

# Check if data needs to be uploaded
if [ -f "processed_abstracts.jsonl" ]; then
    echo ""
    echo "📚 Uploading medical data to Weaviate..."
    python upload_to_weaviate.py
    echo "✅ Medical data uploaded"
else
    echo "⚠️  No processed medical data found (processed_abstracts.jsonl)"
    echo "💡 You can add medical abstracts and run: python upload_to_weaviate.py"
fi

# GPU check (optional)
echo ""
echo "🎮 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    
    # Install CUDA support for spaCy if available
    echo "🚀 Installing CUDA support for spaCy..."
    pip install spacy[cuda-autodetect] || echo "⚠️  CUDA support installation failed (optional)"
else
    echo "⚠️  No NVIDIA GPU detected. Will use CPU mode."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start the chatbot:"
echo "   ./start_chatbot.sh"
echo ""
echo "🌐 The web interface will be available at:"
echo "   http://localhost:7871"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Ensure LM Studio is running (if using local models)"
echo "   3. Run ./start_chatbot.sh to start the application"
echo ""
echo "📚 For more information, see README.md"
