#!/usr/bin/env python3
"""
Web-based Speech-to-Speech Demo using FullDuplexSession.

Implements Freeze-Omni style full-duplex conversation:
- Browser-level AEC (echoCancellation: true)
- Separate AudioContexts for input/output
- Interruption support via VAD
- Streaming audio generation

Usage:
    python demo/s2s_web.py
    python demo/s2s_web.py --model mazesmazes/tiny-audio-s2s --port 8000
"""

import argparse
import asyncio
import os
from typing import Optional

import numpy as np
import scipy.signal
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Tiny Audio S2S Demo")

# Global model (initialized on startup)
model = None

# Audio settings
OUTPUT_SAMPLE_RATE = 24000  # Mimi native
BROWSER_SAMPLE_RATE = 48000  # Browser playback

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses brief - 1-2 sentences. Be friendly and natural."""

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tiny Audio S2S</title>
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
        .status.idle { background: #f5f5f5; color: #666; }
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
        }
        button:disabled { opacity: 0.5; }
        #startBtn { background: #007bff; color: white; }
        #stopBtn { background: #dc3545; color: white; }
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
        .user { color: #007bff; }
        .assistant { color: #28a745; }
        .system { color: #999; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tiny Audio S2S</h1>
        <div id="status" class="status idle">Click Start</div>
        <div>
            <button id="startBtn" onclick="start()">Start</button>
            <button id="stopBtn" onclick="stop()" disabled>Stop</button>
        </div>
        <div class="transcript" id="transcript"></div>
    </div>
    <script>
        let ws, inputCtx, outputCtx, stream, processor;
        let audioQueue = [], playing = false;

        async function start() {
            // Mic with browser AEC (Freeze-Omni style)
            stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false }
            });

            // Separate contexts for input/output (Freeze-Omni style)
            inputCtx = new AudioContext({ sampleRate: 16000 });
            outputCtx = new AudioContext({ sampleRate: 48000 });

            const src = inputCtx.createMediaStreamSource(stream);
            processor = inputCtx.createScriptProcessor(512, 1, 1);

            ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                processor.onaudioprocess = e => {
                    if (ws.readyState !== 1) return;
                    const f = e.inputBuffer.getChannelData(0);
                    const i = new Int16Array(f.length);
                    for (let j = 0; j < f.length; j++) i[j] = Math.max(-32768, Math.min(32767, f[j] * 32768));
                    ws.send(i.buffer);
                };
                src.connect(processor);
                processor.connect(inputCtx.destination);
            };

            ws.onmessage = e => {
                if (typeof e.data === 'string') {
                    const m = JSON.parse(e.data);
                    if (m.type === 'state') setStatus(m.state);
                    else if (m.type === 'text') addText('assistant', m.content);
                    else if (m.type === 'interrupted') { audioQueue = []; playing = false; addText('system', '[interrupted]'); }
                } else {
                    audioQueue.push(e.data);
                    if (!playing) playNext();
                }
            };

            ws.onclose = () => cleanup();
        }

        function setStatus(s) {
            const el = document.getElementById('status');
            el.className = 'status ' + s;
            el.textContent = s.charAt(0).toUpperCase() + s.slice(1);
        }

        function addText(role, text) {
            const p = document.createElement('p');
            p.className = role;
            p.textContent = (role === 'assistant' ? 'Assistant: ' : '') + text;
            document.getElementById('transcript').appendChild(p);
        }

        async function playNext() {
            if (!audioQueue.length) { playing = false; return; }
            playing = true;
            const buf = audioQueue.shift();
            const i16 = new Int16Array(buf), f32 = new Float32Array(i16.length);
            for (let j = 0; j < i16.length; j++) f32[j] = i16[j] / 32768;
            const ab = outputCtx.createBuffer(1, f32.length, 48000);
            ab.getChannelData(0).set(f32);
            const s = outputCtx.createBufferSource();
            s.buffer = ab;
            s.connect(outputCtx.destination);
            s.onended = playNext;
            s.start();
        }

        function stop() { if (ws) ws.close(); cleanup(); }

        function cleanup() {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            if (processor) processor.disconnect();
            if (stream) stream.getTracks().forEach(t => t.stop());
            if (inputCtx) inputCtx.close();
            if (outputCtx) outputCtx.close();
            audioQueue = []; playing = false;
        }
    </script>
</body>
</html>
"""


def load_model(model_id: str):
    """Load model with audio head."""
    global model
    from tiny_audio.asr_modeling import ASRModel

    print(f"Loading model: {model_id}")
    model = ASRModel.from_pretrained(model_id)
    model.eval()
    model.system_prompt = SYSTEM_PROMPT

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else None
    model = model.to(device=device, dtype=dtype) if dtype else model.to(device)

    print(f"Model on {device}")

    if model.audio_head:
        model.audio_head.load_mimi_decoder(device=device)
        print("Mimi decoder loaded")

    model.load_vad()
    print("VAD loaded")


def resample_to_browser(audio: np.ndarray) -> bytes:
    """Resample from Mimi (24kHz) to browser (48kHz)."""
    mx = max(abs(audio.min()), abs(audio.max()))
    if mx > 0:
        audio = audio / mx * 0.9
    resampled = scipy.signal.resample(audio, int(len(audio) * BROWSER_SAMPLE_RATE / OUTPUT_SAMPLE_RATE))
    return (np.clip(resampled, -1, 1) * 32767).astype(np.int16).tobytes()


@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket handler using FullDuplexSession."""
    await websocket.accept()

    from tiny_audio.full_duplex import FullDuplexSession, FullDuplexConfig, ConversationState

    # Async-safe message sending
    loop = asyncio.get_event_loop()
    send_lock = asyncio.Lock()

    async def safe_send_json(data):
        async with send_lock:
            try:
                await websocket.send_json(data)
            except Exception:
                pass

    async def safe_send_bytes(data):
        async with send_lock:
            try:
                await websocket.send_bytes(data)
            except Exception:
                pass

    # Callbacks (called from background threads)
    def on_state(state: ConversationState):
        asyncio.run_coroutine_threadsafe(
            safe_send_json({"type": "state", "state": state.value}),
            loop
        )

    def on_text(text: str, interim: bool):
        asyncio.run_coroutine_threadsafe(
            safe_send_json({"type": "text", "content": text, "interim": interim}),
            loop
        )

    def on_audio(audio: torch.Tensor):
        audio_bytes = resample_to_browser(audio.squeeze().cpu().numpy())
        asyncio.run_coroutine_threadsafe(
            safe_send_bytes(audio_bytes),
            loop
        )

    def on_interrupted():
        asyncio.run_coroutine_threadsafe(
            safe_send_json({"type": "interrupted"}),
            loop
        )

    # Create session with callbacks
    session = FullDuplexSession(
        model=model,
        config=FullDuplexConfig(
            vad_threshold=0.5,
            silence_duration_ms=700,
            audio_chunk_size=4,
        ),
        on_state_change=on_state,
        on_text=on_text,
        on_audio=on_audio,
        on_interrupted=on_interrupted,
    )

    session.start()

    try:
        while True:
            data = await websocket.receive_bytes()
            audio = np.frombuffer(data, dtype=np.int16)
            session.push_audio(audio)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mazesmazes/tiny-audio-s2s")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    load_model(args.model)
    print(f"\nServer at http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
