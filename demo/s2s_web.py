#!/usr/bin/env python3
"""
Web-based Speech-to-Speech Demo using native Mimi audio decoding.

Uses WebSocket for real-time audio streaming and Silero VAD for turn detection.
Uses the model's audio_head to generate Mimi codec tokens, then decodes to audio.

Usage:
    python demo/s2s_web.py
    python demo/s2s_web.py --model mazesmazes/tiny-audio-s2s --port 8000

Then open http://localhost:8000 in your browser.

Requirements:
    pip install moshi_mlx  # Mac (recommended)
    # or transformers with Mimi support for other platforms
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

app = FastAPI(title="Tiny Audio S2S Demo")

# Global state (initialized on startup)
model = None
processor = None
vad_model = None
vad_utils = None

# Audio settings
SAMPLE_RATE = 16000
MIMI_SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 48000  # Browser-native rate
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 1400  # End of speech after this much silence

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses brief and conversational - aim for 1-2 sentences. Be friendly and natural. Do not use emojis or special characters."""

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tiny Audio S2S Demo</title>
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
        .subtitle { color: #666; margin-top: -10px; font-size: 14px; }
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
        .transcript .interim { color: #6c757d; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tiny Audio S2S</h1>
        <p class="subtitle">Native Speech-to-Speech with Mimi codec</p>

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

        let interimElement = null;

        function addTranscript(role, text, interim = false) {
            if (interim) {
                if (!interimElement) {
                    interimElement = document.createElement('p');
                    interimElement.className = 'assistant interim';
                    transcriptEl.appendChild(interimElement);
                }
                interimElement.textContent = 'Assistant: ' + text;
            } else {
                if (interimElement) {
                    interimElement.remove();
                    interimElement = null;
                }
                const p = document.createElement('p');
                p.className = role;
                p.textContent = (role === 'user' ? 'You: ' : 'Assistant: ') + text;
                transcriptEl.appendChild(p);
            }
            transcriptEl.scrollTop = transcriptEl.scrollHeight;
        }

        async function start() {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error('getUserMedia not available. Use localhost or HTTPS.');
                }

                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    }
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    setStatus('listening', 'Listening...');
                    isRunning = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;

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
                            addTranscript(msg.role, msg.text, msg.interim || false);
                        }
                    } else {
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
                const int16 = new Int16Array(arrayBuffer);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) {
                    float32[i] = int16[i] / 32768;
                }

                if (!playbackContext || playbackContext.state === 'closed') {
                    playbackContext = new AudioContext({ sampleRate: 48000 });
                }

                const audioBuffer = playbackContext.createBuffer(1, float32.length, 48000);
                audioBuffer.getChannelData(0).set(float32);

                const source = playbackContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(playbackContext.destination);

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


def load_models(model_id: str):
    """Load ASR model with audio head and VAD."""
    global model, processor, vad_model, vad_utils

    print(f"Loading model: {model_id}")

    from tiny_audio.asr_modeling import ASRModel
    from tiny_audio.asr_processing import ASRProcessor

    model = ASRModel.from_pretrained(model_id)
    model.eval()

    # Set system prompt
    model.system_prompt = SYSTEM_PROMPT

    processor = ASRProcessor.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Model loaded on {device}")
    if model.audio_head is not None:
        print("Audio head: enabled (native S2S)")
    else:
        print("WARNING: Model has no audio_head - S2S will not work!")

    # Load Silero VAD
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    print("VAD loaded")


def process_audio(audio_bytes: bytes) -> dict:
    """Process audio with speech-to-speech using Mimi codec.

    Returns dict with text and audio data.
    """
    if len(audio_bytes) == 0 or model is None:
        return {"text": "", "audio": b""}

    # Convert raw PCM bytes to float32 numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Process with ASRProcessor to get mel features
    inputs = processor(
        audio_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    input_features = inputs["input_features"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    else:
        attention_mask = torch.ones(input_features.shape[0], input_features.shape[-1], device=model.device)

    # Generate text + codec tokens
    try:
        result = model.generate_with_audio(
            input_features=input_features,
            audio_attention_mask=attention_mask,
        )

        text = result["text"][0] if result["text"] else ""
        codec_tokens = result["codec_tokens"]

        # Decode codec tokens to audio waveform
        waveform = model.decode_audio(codec_tokens)

        # Convert to numpy
        audio_out = waveform.squeeze().cpu().numpy()

        # Resample from 24kHz to 48kHz for browser playback
        num_samples = int(len(audio_out) * OUTPUT_SAMPLE_RATE / MIMI_SAMPLE_RATE)
        audio_resampled = scipy.signal.resample(audio_out, num_samples)

        # Convert to int16 bytes
        audio_int16 = (np.clip(audio_resampled, -1, 1) * 32767).astype(np.int16)

        return {"text": text, "audio": audio_int16.tobytes()}

    except Exception as e:
        print(f"S2S error: {e}")
        import traceback
        traceback.print_exc()
        return {"text": f"Error: {e}", "audio": b""}


@app.get("/")
async def get_index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    audio_buffer = []
    is_speaking = False
    silence_frames = 0
    frames_per_chunk = 4096
    silence_threshold = int(SILENCE_DURATION_MS / (frames_per_chunk / SAMPLE_RATE * 1000))

    if vad_utils is None:
        await websocket.close(code=1011, reason="VAD not loaded")
        return
    get_speech_timestamps = vad_utils[0]

    try:
        # Send ready message
        await websocket.send_json(
            {"type": "status", "status": "listening", "text": "Ready - start speaking!"}
        )

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
                    is_speaking = False
                    silence_frames = 0

                    if audio_buffer:
                        await websocket.send_json(
                            {"type": "status", "status": "processing", "text": "Processing..."}
                        )

                        full_audio = b"".join(audio_buffer)
                        audio_buffer = []

                        result = process_audio(full_audio)

                        if result["text"]:
                            await websocket.send_json(
                                {"type": "transcript", "role": "assistant", "text": result["text"], "interim": False}
                            )

                        if result["audio"]:
                            await websocket.send_bytes(result["audio"])

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tiny Audio S2S Web Demo")
    parser.add_argument(
        "--model", "-m", default="mazesmazes/tiny-audio-s2s", help="HuggingFace model ID with audio_head"
    )
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    load_models(args.model)

    print(f"\nStarting S2S server at http://localhost:{args.port}")
    print("Open this URL in Chrome/Edge for best echo cancellation.\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
