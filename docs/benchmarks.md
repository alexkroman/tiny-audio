# Benchmarks

## AIR-Bench Foundation (Speech Tasks)

[AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) is a comprehensive benchmark for evaluating audio understanding capabilities across speech, sound, and music tasks.

### Leaderboard

| Category | tiny-audio | Qwen-Audio-Turbo | Qwen-Audio | Whisper+GPT4 | SALMONN | Pandagpt | Next-gpt | BLSP | SpeechGPT |
|----------|------------|------------------|------------|--------------|---------|----------|----------|------|-----------|
| Speech Grounding | 40.0% | 45.4% | 56.1% | 35.0% | 25.3% | 23.0% | 25.4% | 25.0% | 28.8% |
| Spoken Language ID | 60.0% | 95.9% | 92.8% | 96.8% | 28.1% | 34.6% | 23.7% | 30.8% | 39.6% |
| Speaker Gender | **90.0%** | 82.5% | 67.2% | 21.9% | 35.5% | 66.5% | 57.0% | 33.2% | 29.2% |
| Emotion Recognition | 50.0% | 60.0% | 43.2% | 59.5% | 29.9% | 26.0% | 25.7% | 27.4% | 37.6% |
| Speaker Age | 55.0% | 58.8% | 36.0% | 41.1% | 48.7% | 42.5% | 62.4% | 51.2% | 20.4% |
| Speech Entity Recognition | 60.0% | 48.1% | 71.2% | 69.8% | 51.7% | 34.0% | 26.1% | 37.2% | 35.9% |
| Intent Classification | **80.0%** | 56.4% | 77.8% | 87.7% | 36.7% | 28.5% | 25.6% | 46.6% | 45.8% |
| Speaker Number | 30.0% | 54.3% | 35.3% | 30.0% | 34.3% | 43.2% | 25.4% | 28.1% | 32.6% |
| Synthesized Voice Detection | 55.0% | 69.3% | 48.3% | 40.5% | 50.0% | 53.1% | 30.8% | 50.0% | 39.2% |

**Bold** = tiny-audio outperforms all other models on that task.

## MMAU (Speech Test-Mini)

[MMAU](https://github.com/MMAU-Benchmark/MMAU) (Massive Multi-task Audio Understanding) evaluates audio models on speech, sound, and music understanding. Below are the Speech test-mini scores.

### Leaderboard

| Model | Size | Speech |
|-------|------|--------|
| Nova 2 Omni | - | 81.98 |
| Audio-Thinker | 8.4B | 76.88 |
| Gemini 2.5 Flash | - | 76.58 |
| Step-Audio-2 | - | 75.15 |
| Gemini 2.0 Flash | - | 75.08 |
| Gemini 2.5 Flash Lite | - | 72.07 |
| Gemini 2.5 Pro | - | 71.47 |
| DeSTA2.5-Audio | 8B | 71.47 |
| Qwen2.5-Omni | 8.2B | 70.60 |
| GPT-4o mini Audio | - | 69.07 |
| MiMo-Audio | 7B | 68.17 |
| Step-Audio-2-mini | 8.3B | 68.16 |
| Phi-4-multimodal | 5.5B | 67.27 |
| GPT-4o Audio | - | 66.67 |
| Audio Flamingo 3 | 8.2B | 66.37 |
| Audio Reasoner | 8.2B | 66.07 |
| Kimi-Audio | 8.2B | 62.16 |
| Gemma 3n | 4B | 61.26 |
| **tiny-audio** | **~3B** | **56.40** |
| Qwen2-Audio-Instruct | 7B | 55.26 |
| Gemma 3n | 2B | 52.22 |
| Audio Flamingo 2 | 3B | 44.74 |
| M2UGen | 7B | 33.33 |
| MusiLingo | 7B | 31.23 |
| SALMONN | 13B | 26.43 |
| MuLLaMa | 7B | 17.42 |
| LTU | 7B | 15.92 |
| GAMA | 7B | 12.91 |
| GAMA-IT | 7B | 10.81 |
| Audio Flamingo Chat | 1B | 6.91 |
