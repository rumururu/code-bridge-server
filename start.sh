#!/bin/bash
# Code Bridge Server Startup Script

cd "$(dirname "$0")"

# Configuration
PORT=8080
USE_TUNNEL=${USE_TUNNEL:-false}  # Set USE_TUNNEL=true to enable Cloudflare Tunnel

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Code Bridge Server...${NC}"

# Start Python server
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start server in background if tunnel is enabled
if [ "$USE_TUNNEL" = "true" ]; then
    echo -e "${YELLOW}Starting with Cloudflare Tunnel...${NC}"

    # Start Python server in background
    uvicorn main:app --host 0.0.0.0 --port $PORT &
    SERVER_PID=$!

    # Wait for server to start
    sleep 2

    # Start Cloudflare Tunnel
    echo -e "${GREEN}Starting Cloudflare Tunnel...${NC}"
    cloudflared tunnel --url http://localhost:$PORT

    # Cleanup on exit
    kill $SERVER_PID 2>/dev/null
else
    # Local only
    echo -e "${GREEN}Local mode: http://localhost:$PORT${NC}"
    echo -e "${YELLOW}To enable external access, run: USE_TUNNEL=true ./start.sh${NC}"
    uvicorn main:app --host 0.0.0.0 --port $PORT
fi
