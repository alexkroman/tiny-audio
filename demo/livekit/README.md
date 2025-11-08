# LiveKit Voice Agent with Tiny Audio ASR + TTS

This demo implements a real-time voice agent using LiveKit that:
- Listens to participant audio
- Processes it with the Tiny Audio ASR model using the "continue" task
- Generates a response with integrated TTS
- Plays the TTS audio back to the participant

## Features

- 🎙️ **Real-time speech processing** with Voice Activity Detection (VAD)
- 🤖 **Integrated ASR + TTS** - Speech goes in, speech comes out
- 🚀 **GPU-accelerated** when available (CUDA or Apple Silicon)
- 🔄 **Continuous conversation** using the "continue" task
- 🎯 **Low latency** with optimized buffering

## Prerequisites

1. **Python 3.8+**
2. **LiveKit Server** running locally or remotely
3. **Model Requirements**:
   - The Tiny Audio model with TTS support
   - ~10GB disk space for model weights
   - 8GB+ RAM (16GB recommended)
   - GPU recommended for best performance

## Installation

1. **Clone and navigate to the demo**:
   ```bash
   cd demo/livekit
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the agent**:
   ```bash
   cp .env.example .env
   # Edit .env with your LiveKit credentials
   ```

## Configuration

Edit `.env` file with your settings:

```env
# LiveKit server configuration
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Model configuration
MODEL_PATH=mazesmazes/tiny-audio  # or local path
USE_LOCAL_MODEL=false

# TTS configuration
TTS_VOICE=af_heart  # Voice selection
TTS_SPEED=1.0      # Speech speed (0.5-2.0)

# Debug
DEBUG=false
```

### Available TTS Voices

**American English**:
- Female: `af_heart`, `af_bella`, `af_sarah`, `af_nicole`, `af_sky`
- Male: `am_adam`, `am_michael`

**British English**:
- Female: `bf_emma`, `bf_isabella`
- Male: `bm_george`, `bm_lewis`

## Running the Agent

### Method 1: Using the launch script (recommended)

```bash
./run_agent.sh
```

### Method 2: Manual execution

```bash
# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Run the agent
python voice_agent.py \
    --url "$LIVEKIT_URL" \
    --api-key "$LIVEKIT_API_KEY" \
    --api-secret "$LIVEKIT_API_SECRET"
```

## Testing the Agent

### Using LiveKit Playground

1. Go to [LiveKit Playground](https://playground.livekit.io/)
2. Configure with your server URL and generate a token
3. Join a room
4. The agent will automatically connect when participants join

### Using LiveKit CLI

```bash
# Install LiveKit CLI
brew install livekit-cli  # macOS
# or download from https://github.com/livekit/livekit-cli

# Create a room
livekit-cli create-room --url $LIVEKIT_URL \
    --api-key $LIVEKIT_API_KEY \
    --api-secret $LIVEKIT_API_SECRET \
    my-room

# Generate a token
livekit-cli create-token --url $LIVEKIT_URL \
    --api-key $LIVEKIT_API_KEY \
    --api-secret $LIVEKIT_API_SECRET \
    --identity user1 \
    --room my-room
```

## How It Works

1. **Audio Reception**: The agent subscribes to participant audio tracks
2. **Buffering**: Audio is buffered until VAD detects speech
3. **ASR Processing**: The model processes audio with the "continue" task
4. **TTS Generation**: The model generates TTS from its response
5. **Audio Playback**: TTS audio is sent back through LiveKit

### Data Flow

```
Participant Audio (LiveKit)
    ↓
Audio Buffering + VAD
    ↓
ASR Model (continue task)
    ↓
Text Response
    ↓
Integrated TTS (on GPU)
    ↓
Audio Output (LiveKit)
```

## Performance Optimization

### GPU Acceleration

The agent automatically detects and uses:
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS)
- CPU fallback

### Memory Management

- Model runs in inference mode (eval)
- TTS runs on same device as ASR
- Efficient audio buffering

### Latency Optimization

- Minimal audio buffering (0.5s default)
- Direct TTS generation without external API
- GPU-accelerated processing

## Troubleshooting

### Agent doesn't connect
- Check LiveKit server is running
- Verify credentials in `.env`
- Check firewall/network settings

### No audio output
- Ensure model has TTS enabled
- Check TTS voice configuration
- Verify audio track publication

### High latency
- Use GPU if available
- Reduce `max_new_tokens` in voice_agent.py
- Adjust buffer size

### Memory issues
- Use smaller model variant
- Enable gradient checkpointing
- Reduce batch size

## Advanced Configuration

### Custom Model Path

To use a local model:
```env
MODEL_PATH=/path/to/your/model
USE_LOCAL_MODEL=true
```

### Adjusting Response Length

Edit `voice_agent.py`:
```python
result = self.model.generate_with_tts(
    ...
    max_new_tokens=100,  # Shorter responses
    num_beams=2,         # Faster generation
)
```

### Custom VAD Settings

Modify VAD sensitivity in `voice_agent.py`:
```python
self.vad = silero.VAD.load(
    threshold=0.5,  # Adjust sensitivity
)
```

## Architecture

```
LiveKitVoiceAgent
    ├── Audio Input Handler
    │   ├── Track Subscription
    │   ├── Audio Buffering
    │   └── VAD Processing
    │
    ├── TinyAudioVoiceAgent
    │   ├── Model Loading
    │   ├── ASR Processing
    │   └── TTS Generation
    │
    └── Audio Output Handler
        ├── Audio Source
        ├── Track Publication
        └── Frame Transmission
```

## Development

### Running in Debug Mode

```env
DEBUG=true
```

### Custom Processing

Extend `TinyAudioVoiceAgent.process_audio()` for custom logic:
```python
async def process_audio(self, audio_data):
    # Custom preprocessing
    audio_data = custom_filter(audio_data)

    # Process with model
    result = await super().process_audio(audio_data)

    # Custom postprocessing
    return custom_enhance(result)
```

## License

This demo is part of the Tiny Audio project. See main LICENSE file.