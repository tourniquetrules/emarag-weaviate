# Daemon Mode Usage

The Emergency Medicine RAG Chatbot can now run in the background as a daemon process, allowing you to close the terminal while keeping the chatbot running.

## Quick Start

```bash
# Start the chatbot in background
./start_chatbot_daemon.sh start

# Check status
./start_chatbot_daemon.sh status

# Stop the chatbot
./start_chatbot_daemon.sh stop
```

## Available Commands

| Command | Description |
|---------|-------------|
| `start` | Start the chatbot in background |
| `stop` | Stop the chatbot |
| `restart` | Restart the chatbot |
| `status` | Show chatbot status and uptime |
| `logs` | Show recent logs (last 50 lines) |
| `follow` | Follow logs in real-time (Ctrl+C to exit) |

## Background Operation

When running in daemon mode:
- ✅ **Process runs in background** - you can close the terminal
- ✅ **Automatic logging** - all output saved to `chatbot.log`
- ✅ **PID management** - process ID saved to `chatbot.pid`
- ✅ **Graceful shutdown** - handles SIGTERM and SIGINT signals
- ✅ **Status monitoring** - check uptime and process health
- ✅ **Auto-restart protection** - prevents multiple instances

## Examples

```bash
# Start and check status
./start_chatbot_daemon.sh start
./start_chatbot_daemon.sh status

# View logs
./start_chatbot_daemon.sh logs

# Follow logs in real-time
./start_chatbot_daemon.sh follow

# Restart if needed
./start_chatbot_daemon.sh restart

# Stop when done
./start_chatbot_daemon.sh stop
```

## Files Created

- `chatbot.pid` - Process ID file (excluded from git)
- `chatbot.log` - Application logs (excluded from git)

## Web Interface

The chatbot web interface remains available at:
- **Local**: http://localhost:7871
- **Cloudflare Tunnel**: https://weviate.haydd.com (if configured)

## Troubleshooting

If the chatbot fails to start:
1. Check logs: `./start_chatbot_daemon.sh logs`
2. Verify Weaviate is running: `docker-compose ps`
3. Check port availability: `netstat -tlnp | grep :7871`
4. Restart Weaviate if needed: `docker-compose restart`
