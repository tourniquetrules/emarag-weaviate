#!/bin/bash

# Emergency Medicine RAG Chatbot Background Startup Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIDFILE="$SCRIPT_DIR/chatbot.pid"
LOGFILE="$SCRIPT_DIR/chatbot.log"

# Function to check if chatbot is running
is_running() {
    [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null
}

# Function to start the chatbot
start_chatbot() {
    if is_running; then
        echo "🏥 Emergency Medicine RAG Chatbot is already running (PID: $(cat $PIDFILE))"
        echo "🌐 Web interface available at: http://localhost:7871"
        exit 0
    fi

    echo "🏥 Starting Emergency Medicine RAG Chatbot in background..."
    echo "==========================================="

    # Check if virtual environment exists
    if [ ! -d "venv312" ]; then
        echo "❌ Virtual environment not found. Please run setup.sh first."
        exit 1
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
    echo "🚀 Launching Emergency Medicine RAG Chatbot in background..."
    echo "📝 Logs will be written to: $LOGFILE"
    echo "🌐 Web interface will be available at: http://localhost:7871"
    echo ""

    # Start the chatbot in background with nohup
    source venv312/bin/activate
    nohup python medical_rag_chatbot.py > "$LOGFILE" 2>&1 &
    
    # Save PID
    echo $! > "$PIDFILE"
    
    # Wait a moment to see if it starts successfully
    sleep 3
    if is_running; then
        echo "✅ Chatbot started successfully (PID: $(cat $PIDFILE))"
        echo "📊 Use './start_chatbot_daemon.sh status' to check status"
        echo "⏹️  Use './start_chatbot_daemon.sh stop' to stop the chatbot"
        echo "📝 Use './start_chatbot_daemon.sh logs' to view logs"
    else
        echo "❌ Failed to start chatbot. Check logs: $LOGFILE"
        rm -f "$PIDFILE"
        exit 1
    fi
}

# Function to stop the chatbot
stop_chatbot() {
    if ! is_running; then
        echo "🏥 Emergency Medicine RAG Chatbot is not running"
        rm -f "$PIDFILE"
        exit 0
    fi

    echo "⏹️  Stopping Emergency Medicine RAG Chatbot..."
    kill $(cat "$PIDFILE")
    
    # Wait for graceful shutdown
    timeout=10
    elapsed=0
    while is_running && [ $elapsed -lt $timeout ]; do
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
    done
    
    # Force kill if still running
    if is_running; then
        echo ""
        echo "⚡ Force stopping chatbot..."
        kill -9 $(cat "$PIDFILE")
    fi
    
    rm -f "$PIDFILE"
    echo ""
    echo "✅ Chatbot stopped successfully"
}

# Function to show status
show_status() {
    if is_running; then
        echo "🏥 Emergency Medicine RAG Chatbot is running (PID: $(cat $PIDFILE))"
        echo "🌐 Web interface: http://localhost:7871"
        echo "📝 Log file: $LOGFILE"
        echo "⏱️  Uptime: $(ps -o etime= -p $(cat $PIDFILE) 2>/dev/null | xargs)"
    else
        echo "🏥 Emergency Medicine RAG Chatbot is not running"
        rm -f "$PIDFILE"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOGFILE" ]; then
        echo "📝 Showing chatbot logs (last 50 lines):"
        echo "========================================"
        tail -50 "$LOGFILE"
    else
        echo "📝 No log file found at: $LOGFILE"
    fi
}

# Function to follow logs
follow_logs() {
    if [ -f "$LOGFILE" ]; then
        echo "📝 Following chatbot logs (Ctrl+C to exit):"
        echo "==========================================="
        tail -f "$LOGFILE"
    else
        echo "📝 No log file found at: $LOGFILE"
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        start_chatbot
        ;;
    stop)
        stop_chatbot
        ;;
    restart)
        stop_chatbot
        sleep 2
        start_chatbot
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    follow)
        follow_logs
        ;;
    *)
        echo "🏥 Emergency Medicine RAG Chatbot Daemon"
        echo "======================================="
        echo "Usage: $0 {start|stop|restart|status|logs|follow}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the chatbot in background"
        echo "  stop    - Stop the chatbot"
        echo "  restart - Restart the chatbot"
        echo "  status  - Show chatbot status"
        echo "  logs    - Show recent logs"
        echo "  follow  - Follow logs in real-time"
        echo ""
        exit 1
        ;;
esac
