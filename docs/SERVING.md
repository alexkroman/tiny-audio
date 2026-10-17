# Serving tiny-audio over HTTP

`ta serve` puts the model behind a batched HTTP server: requests arriving together share GPU
batches, so throughput grows with load (about 460x real time at 128 concurrent requests on an RTX
4090). To run it on a RunPod GPU:

```bash
poetry run ta runpod up --serve                 # create an inference pod; prints <POD_ID>
poetry run ta runpod wait <POD_ID>              # prints <HOST> <PORT>
poetry run ta runpod deploy <HOST> <PORT>       # sync the project, install the fast kernels
TINY_AUDIO_API_KEY=my-secret poetry run ta runpod serve <HOST> <PORT> --no-attach
# Ready when https://<POD_ID>-8000.proxy.runpod.net/health answers (a few minutes: it compiles first)
```

Without `TINY_AUDIO_API_KEY` the server is open to anyone who has the URL. `ta serve` also runs
locally, on CUDA, Apple Silicon, or CPU.

Send the audio as the request body, with options in the query string:

```bash
curl -X POST "https://<POD_ID>-8000.proxy.runpod.net/?return_timestamps=true" \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

```python
import httpx

response = httpx.post(
    "https://<POD_ID>-8000.proxy.runpod.net/",
    params={"return_speakers": "true", "num_speakers": "2"},
    content=open("meeting.wav", "rb").read(),
    headers={"Authorization": "Bearer my-secret"},
    timeout=600,
)
print(response.json()["text"])
```

- **Options:** `return_timestamps`, `return_speakers`, `num_speakers` and `max_speakers`, as in the
  pipeline. The response is the same dict the pipeline returns.
- **JSON body:** to send JSON instead, use `{"inputs": "<base64 audio>", "parameters": {...}}`.
- **Audio formats:** anything FFmpeg can read.
- **Errors:** `400` with `{"error": ...}` for bad audio or options, and `401` for a wrong key.
- **Long requests:** `POST /jobs` takes the same body and options but answers `202` with
  `{"id": ...}` at once; poll `GET /jobs/<id>`, which answers `202` while the job runs and then the
  response `POST /` would have given. Use it for anything that may run past the proxy's 100 s limit,
  such as a whole meeting with `return_speakers`; `ta eval --endpoint` does this for speaker
  datasets.
- **Other endpoints:** `GET /health` and `GET /stats` (batch sizes and GPU time).

RunPod's HTTP proxy rejects request bodies over 500 MiB, and it drops any request that takes more
than 100 seconds. For long recordings, send 16 kHz mono FLAC:

```bash
ffmpeg -i recording.wav -ac 1 -ar 16000 recording.flac
```

That's about 1 MB per minute of audio, and it costs nothing in accuracy, because the server converts
everything to 16 kHz mono anyway. On an RTX 4090, 45 minutes of audio takes about 10 seconds, or 21
seconds with speaker labels. The [demo Space](../demo/app.py) calls the server this way.
