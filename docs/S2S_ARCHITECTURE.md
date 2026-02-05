# Speech-to-Speech Model: Simple Overview

## What You Can Do With It

**Speech-to-Speech**: Speak naturally and get spoken responses back

**Speech-to-Text**: Transcribe spoken audio into written text

**Text-to-Speech**: Type text and hear it spoken with natural intonation

**Text-to-Text**: Have regular text conversations like a standard chatbot

## How It Works

The pipeline has 6 main steps:

1. **Listen** — Your voice goes in (audio recording)
1. **Understand** — Whisper encoder converts speech to representations
1. **Project** — MLP projector bridges audio representations to the language model's embedding space
1. **Think** — A language model (LLM) processes what you said and decides what to say back
1. **Generate Speech Codes** — An "Audio Head" creates special audio codes that represent the response
1. **Speak** — Mimi decoder turns those codes into actual audio you can hear

## What Gets Trained vs. Frozen

**Trained (learns during training):**

- The bridge between audio and language (MLP Projector)
- The bridge between language and speaking (Transformer + Depformer) - this is the part that generates speech

**Frozen (stays the same):**

- Whisper (the listening part)
- The LLM (the thinking part)
- Mimi decoder (the final audio generation)

## Training Data

The model learns from diverse examples:

- Transcription tasks (audio → text)
- Question answering (audio question → text answer)
- Conversational responses (casual audio → friendly response)
- Empathetic responses (emotional audio → emotionally aware response)
- Speech descriptions (describing emotion, accent, gender, and pace from audio)

This variety helps the model understand different contexts and respond appropriately.
