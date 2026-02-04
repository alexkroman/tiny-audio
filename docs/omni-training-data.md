# Omni Training Data Examples

This document shows examples of the different training data formats used in `+experiments=omni` training. The model learns from four distinct task types.

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

______________________________________________________________________
