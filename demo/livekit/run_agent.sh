#!/bin/bash

# Tiny Audio LiveKit Voice Agent Launcher

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎙️  Tiny Audio LiveKit Voice Agent${NC}"
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo -e "${YELLOW}Creating .env from .env.example${NC}"
        cp .env.example .env
        echo -e "${RED}Please edit .env with your LiveKit credentials${NC}"
        exit 1
    else
        echo -e "${RED}No .env file found${NC}"
        exit 1
    fi
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check for required environment variables
if [ -z "$LIVEKIT_URL" ] || [ -z "$LIVEKIT_API_KEY" ] || [ -z "$LIVEKIT_API_SECRET" ]; then
    echo -e "${RED}Missing required LiveKit configuration in .env${NC}"
    echo "Please set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Add parent directory to PYTHONPATH for model imports
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(pwd)))"

echo -e "${GREEN}Starting agent...${NC}"
echo "Configuration:"
echo "  LiveKit URL: $LIVEKIT_URL"
echo "  Model: ${MODEL_PATH:-mazesmazes/tiny-audio}"
echo "  TTS Voice: ${TTS_VOICE:-af_heart}"
echo "  TTS Speed: ${TTS_SPEED:-1.0}"
echo ""
echo "Press Ctrl+C to stop the agent"
echo "=================================="

# Run the agent
python voice_agent.py \
    --url "$LIVEKIT_URL" \
    --api-key "$LIVEKIT_API_KEY" \
    --api-secret "$LIVEKIT_API_SECRET"