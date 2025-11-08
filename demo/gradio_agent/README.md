# Gradio Voice Agent Demo

An interactive web-based voice agent powered by the Tiny Audio model, providing real-time speech recognition, conversation continuation, and text-to-speech synthesis.

## Features

- **Real-time Voice Recording**: Record audio directly from your browser
- **Multiple Tasks**: Support for "continue", "transcribe", and "translate" tasks
- **Text-to-Speech**: Generate natural-sounding responses with multiple voice options
- **Continuous Conversation**: Maintain context across multiple interactions
- **Conversation History**: Track and review the full conversation
- **Customizable Settings**: Adjust TTS voice, speed, response length, and generation parameters

## Installation

1. **Install dependencies**:
```bash
cd demo/gradio_agent
pip install -r requirements.txt
```

2. **Ensure the model is available**:
   - The demo will automatically download the model from HuggingFace Hub
   - Or set `use_local_model=True` in the code to use a local model

## Usage

### Basic Usage

1. **Start the application**:
```bash
python app.py
```

2. **Open your browser**:
   - Navigate to `http://localhost:7860`
   - The interface will load automatically

3. **Load the model**:
   - Click the "Load Model" button
   - Wait for the confirmation message

4. **Start talking**:
   - Click the microphone icon to record
   - Speak your message
   - Click stop to process
   - The agent will respond with both text and speech

### Continuous Conversation Mode

The continuous conversation mode allows for natural back-and-forth dialogue:

1. Use the "Continuous Recording" section
2. Enable microphone access
3. Speak naturally - the agent will respond using the "continue" task
4. The conversation maintains context automatically

### Task Modes

- **Continue**: Generates contextual responses to continue the conversation
- **Transcribe**: Converts speech to text without generation
- **Translate**: Translates speech to another language

### Customization

#### TTS Settings
- **Voice**: Choose from 12 different voice options
  - Female voices: af, af_bella, af_nicole, af_heart, af_sarah, bf_emma, bf_isabella
  - Male voices: am_adam, am_michael, bm_george, bm_lewis
- **Speed**: Adjust speech rate from 0.5x to 2.0x

#### Advanced Settings
- **Max Response Length**: Control how long responses can be (10-500 tokens)
- **Temperature**: Adjust creativity/randomness (0.1-2.0)

## Configuration

You can modify the default settings in the `AgentConfig` class:

```python
@dataclass
class AgentConfig:
    model_path: str = "mazesmazes/tiny-audio"
    use_local_model: bool = False
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    max_new_tokens: int = 100
    temperature: float = 0.7
    num_beams: int = 1
```

## Environment Variables

You can also configure the agent using environment variables:

```bash
export MODEL_PATH="mazesmazes/tiny-audio"
export USE_LOCAL_MODEL="false"
export TTS_VOICE="af_heart"
export TTS_SPEED="1.0"
```

## API Integration

The voice agent can be integrated into other applications:

```python
from app import VoiceAgent, AgentConfig

# Initialize agent
config = AgentConfig()
agent = VoiceAgent(config)
agent.load_model()

# Process audio
sample_rate = 16000
audio_data = np.array([...])  # Your audio data

text, tts_audio, history = agent.process_audio(
    (sample_rate, audio_data),
    task="continue"
)
```

## Troubleshooting

### Audio Issues
- Ensure microphone permissions are granted in your browser
- Check that your microphone is working and not muted
- Try refreshing the page if audio stops working

### Model Loading
- Ensure you have sufficient RAM (at least 8GB recommended)
- Check internet connection for model download
- Verify CUDA is available if using GPU

### Performance
- Use fewer beams (num_beams=1) for faster responses
- Reduce max_new_tokens for shorter responses
- Consider using a GPU for better performance

## Development

### Running with custom settings

```bash
# With environment variables
MODEL_PATH="./models/tiny-audio" python app.py

# With different port
python app.py --server-port 8080

# With public sharing link
python app.py --share
```

### Extending the agent

The `VoiceAgent` class can be extended with additional features:

```python
class CustomVoiceAgent(VoiceAgent):
    def process_audio(self, audio_input, task, conversation_context):
        # Add custom preprocessing
        audio_input = self.preprocess(audio_input)

        # Call parent method
        result = super().process_audio(
            audio_input, task, conversation_context
        )

        # Add custom postprocessing
        return self.postprocess(result)
```

## License

This demo is provided as part of the Tiny Audio project. See the main project LICENSE for details.

## Support

For issues or questions:
- Check the main [Tiny Audio repository](https://github.com/mazesmazes/tiny-audio)
- Open an issue with the `demo` tag
- Review existing documentation in the main README