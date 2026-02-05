# Training Data Examples

This document shows examples of the different training data formats used for model training.

## Example 1: Transcription Task

**Audio**: A person saying "The weather is beautiful today."

**Training Format**:

```
User: <audio><audio><audio>... Transcribe:
Assistant: the weather is beautiful today.
```

**Source Datasets**: LoquaciousSet, AMI

______________________________________________________________________

## Example 2: Answer Extraction Task

**Audio**: Someone asking "What is the capital of France?"

**Training Format**:

```
User: <audio><audio><audio>...
Assistant: paris
```

**Source Dataset**: HeySQuAD

______________________________________________________________________

## Example 3: Conversational Response

**Audio**: A happy woman saying "I got the job!"

**Training Format**:

```
User: <audio><audio><audio>...
Assistant: That's wonderful news! Congratulations on your new position!
```

**Source Dataset**: mazesmazes/sift-audio-2

**Note**: No instruction prompt - model learns to respond conversationally to what it hears.

______________________________________________________________________

## Example 4: Empathetic Response

**Audio**: A sad man saying "I lost my keys again."

**Training Format**:

```
User: <audio><audio><audio>...
Assistant: Oh no, that's frustrating! Have you checked your coat pockets?
```

**Source Dataset**: mazesmazes/sift-audio-2

**Note**: Model learns to respond with awareness of speaker's emotional tone.

______________________________________________________________________

## Example 5: Audio Description

**Audio**: An angry young adult male saying "Leave me alone!"

**Training Format**:

```
User: <audio><audio><audio>... Describe all information you can hear:
Assistant: Sounds like an angry young male saying "leave me alone" in a harsh, frustrated voice.
```

**Source Dataset**: mazesmazes/sift-audio-2

**Note**: This mode includes an instruction prompt, teaching the model to describe audio on demand.

______________________________________________________________________

## Example 6: Speech-to-Speech (S2S)

**Audio**: A person saying "The weather is beautiful today."

**Training Format**:

```
User: <audio><audio><audio>...
Assistant: the weather is beautiful today.
```

**Codec Targets** (parallel supervision for Audio Head):

```
codec_targets: [8, T]  # 8 Mimi codebooks × T frames

Codebook 0 (semantic):  [1842, 1203, 892, 1456, ...]   # AR decoder target
Codebook 1 (acoustic):  [456, 1823, 234, 901, ...]    # Depformer target
Codebook 2 (acoustic):  [78, 1234, 567, 2001, ...]    # Depformer target
...
Codebook 7 (acoustic):  [234, 890, 1567, 432, ...]    # Depformer target
```

**Source Dataset**: mazesmazes/libritts-mimi

**Note**: S2S training uses two parallel objectives:

1. **Text loss**: LLM learns to generate transcription from audio embeddings
1. **Codec loss**: Audio Head learns to generate Mimi codes from LLM hidden states

The codec targets are pre-computed using the Mimi encoder and stored alongside the audio.

______________________________________________________________________
