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

## SIFT Training Data Generation

The SIFT (Self-Instruction Fine-Tuning) responses in `mazesmazes/sift-audio-2` are generated using an LLM (SmolLM3-3B) with three different modes. The script `scripts/generate_sift_dataset.py` processes audio datasets with paralinguistic metadata (emotion, gender, age, pace, volume, accent) and generates training responses.

### Mode 1: `sift_s` - Semantic Only (Conversational Response)

Teaches the model to respond conversationally to what it hears, like a voice assistant.

**LLM Prompt Used to Generate Response**:

```
System: You are the user's friend. Respond warmly and briefly to what they tell you.
User: i got the job!
```

**Generated Response**: "That's wonderful news! Congratulations on your new position!"

**Final Training Format** (audio tokens replace text):

```
User: <audio><audio><audio>...
Assistant: That's wonderful news! Congratulations on your new position!
```

**Note**: No paralinguistic metadata is included - the model learns purely from the semantic content.

______________________________________________________________________

### Mode 2: `sift_ssp` - Empathetic Response with Tone Awareness

Teaches the model to respond empathetically, considering the speaker's emotional tone.

**LLM Prompt Used to Generate Response**:

```
System: You can hear audio. Respond empathetically to what the person says, being aware of their tone and emotion. Don't describe the audio - just respond to it naturally.
User: <audio><meta>gender: male, emotion: sad</meta><text>i lost my keys again.</text></audio>
```

**Generated Response**: "Oh no, that's frustrating! Have you checked your coat pockets?"

**Final Training Format** (audio tokens replace text):

```
User: <audio><audio><audio>...
Assistant: Oh no, that's frustrating! Have you checked your coat pockets?
```

**Note**: The LLM sees metadata tags during generation, but the final training data only uses audio tokens. This teaches the model to infer emotion from audio features.

______________________________________________________________________

### Mode 3: `sit_ssp` - Audio Description/Analysis

Teaches the model to describe what it hears, including paralinguistic features.

**LLM Prompt Used to Generate Response**:

```
System: Describe audio in one sentence starting with "Sounds like". Include emotion, gender, what they said (quoted), and voice quality. Example: "Sounds like an angry man saying 'leave me alone' in a harsh voice."
User: <audio><meta>age: young adult, gender: male, emotion: angry</meta><text>leave me alone</text></audio>
```

**Generated Response**: "Sounds like an angry young male saying 'leave me alone' in a harsh, frustrated voice."

**Final Training Format** (audio tokens replace text):

```
User: <audio><audio><audio>... Describe all information you can hear:
Assistant: Sounds like an angry young male saying "leave me alone" in a harsh, frustrated voice.
```

**Note**: This mode includes an instruction prompt in the training data, teaching the model to describe audio on demand.

______________________________________________________________________

## Metadata Extraction

The `generate_sift_dataset.py` script extracts metadata from source datasets:

| Field | Source | Normalization |
|-------|--------|---------------|
| `emotion` | CREMA-D, RAVDESS, TESS, SAVEE, ESD, PODCAST, MELD | Normalized to: angry, happy, sad, surprise, neutral, fear, disgust |
| `gender` | All datasets | Normalized to: male, female |
| `age` | CommonVoice | Grouped: teenager, young adult, middle-age adult, senior |
| `pace` | AbstractTTS datasets (speaking_rate) | Converted: slow (\<6.0), normal (6-9), fast (>9.0) |
| `volume` | AbstractTTS datasets (relative_db) | Converted: quiet (\<-16.4dB), loud (>-10dB), or omitted if normal |
| `accent` | CommonVoice | Passed through as-is |

## Audio Context Format

The metadata is formatted into XML-style tags for the LLM:

```
<audio><meta>age: young adult, gender: female, volume: loud, pace: fast, emotion: happy</meta><text>i got the job!</text></audio>
```

Fields appear in order: age, gender, volume, pace, emotion, accent. Missing fields are omitted.

______________________________________________________________________
