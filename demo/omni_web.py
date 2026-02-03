#!/usr/bin/env python3
"""
Web-based Tiny Audio Voice Agent Demo with browser echo cancellation.

Uses WebSocket for real-time audio streaming and Silero VAD for turn detection.
Browser's built-in echo cancellation prevents feedback loops.

Usage:
    python demo/omni_web.py
    python demo/omni_web.py --model mazesmazes/tiny-audio-omni --port 8000

Then open http://localhost:8000 in your browser.
"""

import argparse
import os

import numpy as np
import scipy.signal
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Tiny Audio Voice Agent")

# Global state (initialized on startup)
model = None
pipeline = None
vad_model = None
vad_utils = None
device = None
tts_voice = "af_heart"

# Audio settings
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 48000  # Browser-native rate
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 700  # End of speech after this much silence

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses brief and conversational - aim for 1-2 sentences. Be friendly and natural. Do not use emojis or special characters."""

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tiny Audio Voice Agent</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { margin-top: 0; color: #333; }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: 500;
        }
        .status.disconnected { background: #fee; color: #c00; }
        .status.connected { background: #efe; color: #070; }
        .status.listening { background: #eef; color: #007; }
        .status.processing { background: #ffe; color: #a70; }
        .status.speaking { background: #fef; color: #707; }
        button {
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
            transition: all 0.2s;
        }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        #startBtn { background: #007bff; color: white; }
        #startBtn:hover:not(:disabled) { background: #0056b3; }
        #stopBtn { background: #dc3545; color: white; }
        #stopBtn:hover:not(:disabled) { background: #c82333; }
        .transcript {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            min-height: 100px;
            max-height: 300px;
            overflow-y: auto;
        }
        .transcript p { margin: 8px 0; }
        .transcript .user { color: #007bff; }
        .transcript .assistant { color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tiny Audio Voice Agent</h1>

        <div id="status" class="status disconnected">Disconnected</div>

        <div>
            <button id="startBtn" onclick="start()">Start Conversation</button>
            <button id="stopBtn" onclick="stop()" disabled>Stop</button>
        </div>

        <div class="transcript" id="transcript"></div>
    </div>

    <script>
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let processor = null;
        let isRunning = false;
        let isPlaying = false;

        const statusEl = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const transcriptEl = document.getElementById('transcript');

        function setStatus(status, text) {
            statusEl.className = 'status ' + status;
            statusEl.textContent = text;
        }

        function addTranscript(role, text) {
            const p = document.createElement('p');
            p.className = role;
            p.textContent = (role === 'user' ? 'You: ' : 'Assistant: ') + text;
            transcriptEl.appendChild(p);
            transcriptEl.scrollTop = transcriptEl.scrollHeight;
        }

        async function start() {
            try {
                // Check for secure context (required for getUserMedia)
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error('getUserMedia not available. Make sure you are using localhost or HTTPS.');
                }

                // Get microphone with echo cancellation
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    }
                });

                // Setup audio processing
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                // Connect WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    setStatus('listening', 'Listening...');
                    isRunning = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;

                    // Send audio chunks (pause while playing response)
                    processor.onaudioprocess = (e) => {
                        if (!isRunning || isPlaying || ws.readyState !== WebSocket.OPEN) return;
                        const float32 = e.inputBuffer.getChannelData(0);
                        const int16 = new Int16Array(float32.length);
                        for (let i = 0; i < float32.length; i++) {
                            int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
                        }
                        ws.send(int16.buffer);
                    };

                    source.connect(processor);
                    processor.connect(audioContext.destination);
                };

                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'status') {
                            setStatus(msg.status, msg.text);
                        } else if (msg.type === 'transcript') {
                            addTranscript(msg.role, msg.text);
                        }
                    } else {
                        // Audio data - play it
                        await playAudio(event.data);
                    }
                };

                ws.onclose = () => {
                    setStatus('disconnected', 'Disconnected');
                    cleanup();
                };

                ws.onerror = (err) => {
                    console.error('WebSocket error:', err);
                    setStatus('disconnected', 'Connection error');
                    cleanup();
                };

            } catch (err) {
                console.error('Error starting:', err);
                setStatus('disconnected', 'Error: ' + err.message);
            }
        }

        let playbackContext = null;

        async function playAudio(arrayBuffer) {
            isPlaying = true;
            setStatus('speaking', 'Speaking...');
            try {
                // Decode as 48kHz int16 mono (server already resampled)
                const int16 = new Int16Array(arrayBuffer);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) {
                    float32[i] = int16[i] / 32768;
                }

                // Create context at 48kHz (matches server output)
                if (!playbackContext || playbackContext.state === 'closed') {
                    playbackContext = new AudioContext({ sampleRate: 48000 });
                }

                // Create buffer and play directly
                const audioBuffer = playbackContext.createBuffer(1, float32.length, 48000);
                audioBuffer.getChannelData(0).set(float32);

                const source = playbackContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(playbackContext.destination);

                // Wait for audio to finish before resuming listening
                await new Promise(resolve => {
                    source.onended = resolve;
                    source.start();
                });
            } catch (err) {
                console.error('Error playing audio:', err);
            } finally {
                isPlaying = false;
                setStatus('listening', 'Listening...');
            }
        }

        function stop() {
            isRunning = false;
            if (ws) ws.close();
            cleanup();
        }

        function cleanup() {
            isRunning = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            if (processor) processor.disconnect();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
            if (audioContext) audioContext.close();
            if (playbackContext) playbackContext.close();
            processor = null;
            mediaStream = null;
            audioContext = null;
            playbackContext = null;
        }
    </script>
</body>
</html>
"""


def load_models(model_id: str, voice: str = "af_heart"):
    """Load ASR model and VAD."""
    global model, pipeline, vad_model, vad_utils, device, tts_voice

    tts_voice = voice
    print(f"Loading model: {model_id}")

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load ASR model
    from tiny_audio.asr_modeling import ASRModel
    from tiny_audio.asr_pipeline import ASRPipeline

    model = ASRModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    model.system_prompt = SYSTEM_PROMPT
    pipeline = ASRPipeline(model, device=device)
    print("ASR model loaded (includes Kokoro TTS)")

    # Load Silero VAD
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    print("VAD loaded")


GREETING_TEXT = "Hi there! How can I help you today?"


def generate_greeting() -> tuple[str, bytes]:
    """Generate greeting audio using TTS."""
    if pipeline is None:
        return GREETING_TEXT, b""

    # Generate TTS audio using pipeline method
    tts_result = pipeline.text_to_speech(GREETING_TEXT, voice=tts_voice)
    audio_out = tts_result.get("audio")

    if audio_out is None or len(audio_out) == 0:
        return GREETING_TEXT, b""

    # Resample from 24kHz to 48kHz for browser playback
    num_samples = int(len(audio_out) * OUTPUT_SAMPLE_RATE / TTS_SAMPLE_RATE)
    audio_resampled = scipy.signal.resample(audio_out, num_samples)

    # Convert to int16 bytes
    audio_int16 = (audio_resampled * 32767).astype(np.int16)
    return GREETING_TEXT, audio_int16.tobytes()


async def process_audio(audio_bytes: bytes) -> tuple[str, bytes]:
    """Process audio and return response text and audio."""
    # Convert to numpy
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_array) == 0:
        return "", b""

    # Use pipeline with return_audio=True to get text + TTS in one call
    result = pipeline(
        {"array": audio_array, "sampling_rate": SAMPLE_RATE},
        return_audio=True,
        tts_voice=tts_voice,
    )

    response_text = result.get("text", "").strip()
    audio_out = result.get("audio")

    if not response_text:
        return "", b""

    if audio_out is None or len(audio_out) == 0:
        return response_text, b""

    # Resample from 24kHz to 48kHz for browser playback
    num_samples = int(len(audio_out) * OUTPUT_SAMPLE_RATE / TTS_SAMPLE_RATE)
    audio_resampled = scipy.signal.resample(audio_out, num_samples)

    # Convert to int16 bytes
    audio_int16 = (audio_resampled * 32767).astype(np.int16)
    return response_text, audio_int16.tobytes()


@app.get("/")
async def get_index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    audio_buffer = []
    is_speaking = False
    silence_frames = 0
    frames_per_chunk = 4096  # Match browser's ScriptProcessor buffer size
    silence_threshold = int(SILENCE_DURATION_MS / (frames_per_chunk / SAMPLE_RATE * 1000))

    if vad_utils is None:
        await websocket.close(code=1011, reason="VAD not loaded")
        return
    get_speech_timestamps = vad_utils[0]

    try:
        # Send greeting on connection
        greeting_text, greeting_audio = generate_greeting()
        if greeting_text:
            await websocket.send_json(
                {"type": "transcript", "role": "assistant", "text": greeting_text}
            )
        if greeting_audio:
            await websocket.send_bytes(greeting_audio)

        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Run VAD
            speech_timestamps = get_speech_timestamps(
                torch.from_numpy(audio_chunk),
                vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
            )

            has_speech = len(speech_timestamps) > 0

            if has_speech:
                if not is_speaking:
                    is_speaking = True
                    audio_buffer = []
                audio_buffer.append(data)
                silence_frames = 0
            elif is_speaking:
                audio_buffer.append(data)
                silence_frames += 1

                if silence_frames >= silence_threshold:
                    # End of speech detected
                    is_speaking = False
                    silence_frames = 0

                    if audio_buffer:
                        await websocket.send_json(
                            {"type": "status", "status": "processing", "text": "Processing..."}
                        )

                        # Process audio
                        full_audio = b"".join(audio_buffer)
                        audio_buffer = []

                        response_text, response_audio = await process_audio(full_audio)

                        if response_text:
                            await websocket.send_json(
                                {"type": "transcript", "role": "assistant", "text": response_text}
                            )

                        if response_audio:
                            await websocket.send_bytes(response_audio)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tiny Audio Voice Agent Web Demo")
    parser.add_argument(
        "--model", "-m", default="mazesmazes/tiny-audio-omni", help="HuggingFace model ID"
    )
    parser.add_argument("--voice", "-v", default="af_heart", help="Kokoro TTS voice ID")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    load_models(args.model, args.voice)

    print(f"\nStarting server at http://localhost:{args.port}")
    print("Open this URL in Chrome/Edge for best echo cancellation.\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
