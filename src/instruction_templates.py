"""
Instruction templates for multi-task audio understanding.

Following Kimi-Audio approach:
- 200 variations for ASR task
- 30 variations for other tasks
- Random selection during training to improve instruction following
"""

import random
from typing import Dict, List

# ASR Instruction Templates (100 most important variations)
# Based on Kimi-Audio paper approach - reduced to most diverse and useful
ASR_TEMPLATES: List[str] = [
    # Core direct transcription (10)
    "Transcribe: <audio>",
    "Transcribe the audio: <audio>",
    "Please transcribe: <audio>",
    "Transcribe this audio: <audio>",
    "Transcribe the following audio: <audio>",
    "Please transcribe the audio: <audio>",
    "Transcribe this recording: <audio>",
    "Please transcribe this recording: <audio>",
    "Transcribe the speech: <audio>",
    "Transcribe this speech: <audio>",

    # Question-based (15)
    "What is being said? <audio>",
    "What is the speaker saying? <audio>",
    "What does the speaker say? <audio>",
    "What is said in the audio? <audio>",
    "What words are spoken? <audio>",
    "What is spoken? <audio>",
    "What do you hear? <audio>",
    "What does the person say? <audio>",
    "What is the message? <audio>",
    "What is the content? <audio>",
    "What speech do you hear? <audio>",
    "What is the utterance? <audio>",
    "What is the dialogue? <audio>",
    "What conversation is taking place? <audio>",
    "What is being narrated? <audio>",

    # Conversion requests (10)
    "Convert this speech to text: <audio>",
    "Convert the audio to text: <audio>",
    "Convert to text: <audio>",
    "Turn this speech into text: <audio>",
    "Transform this speech to text: <audio>",
    "Change speech to text: <audio>",
    "Convert speech into written form: <audio>",
    "Transform audio into text: <audio>",
    "Turn the audio into text: <audio>",
    "Convert this to text: <audio>",

    # Writing/documentation (10)
    "Write down what you hear: <audio>",
    "Write what is being said: <audio>",
    "Write the spoken words: <audio>",
    "Write down the speech: <audio>",
    "Write out what you hear: <audio>",
    "Document the speech: <audio>",
    "Note what you hear: <audio>",
    "Record what you hear: <audio>",
    "Note down what is said: <audio>",
    "Record the spoken words: <audio>",

    # Recognition/identification (8)
    "Recognize the speech: <audio>",
    "Identify the words: <audio>",
    "Identify what is spoken: <audio>",
    "Recognize what is said: <audio>",
    "Identify the speech: <audio>",
    "Recognize the spoken words: <audio>",
    "Detect the speech: <audio>",
    "Identify the spoken content: <audio>",

    # Listen-based (7)
    "Listen and transcribe: <audio>",
    "Listen and write what you hear: <audio>",
    "Listen to the audio and transcribe: <audio>",
    "Listen carefully and transcribe: <audio>",
    "Listen and tell me what is said: <audio>",
    "Listen to this and transcribe: <audio>",
    "Listen and write: <audio>",

    # Provide/generate (8)
    "Provide a transcription: <audio>",
    "Generate a transcription: <audio>",
    "Give me the transcription: <audio>",
    "Create a transcription: <audio>",
    "Produce a transcription: <audio>",
    "Make a transcription: <audio>",
    "Supply the transcription: <audio>",
    "Deliver the transcription: <audio>",

    # Text extraction (7)
    "Extract the text: <audio>",
    "Extract the spoken text: <audio>",
    "Extract what is said: <audio>",
    "Pull out the text: <audio>",
    "Extract text from the audio: <audio>",
    "Extract the words: <audio>",
    "Get the text from this audio: <audio>",

    # Detailed/polite requests (8)
    "Please listen to this audio and provide a transcription: <audio>",
    "I need a transcription of this audio: <audio>",
    "Can you transcribe this audio? <audio>",
    "Could you transcribe this? <audio>",
    "Can you tell me what is said? <audio>",
    "Could you tell me what is being said? <audio>",
    "Please tell me what you hear: <audio>",
    "I need you to transcribe this: <audio>",

    # Accuracy-focused (7)
    "Transcribe word for word: <audio>",
    "Give me a word-for-word transcription: <audio>",
    "Provide an exact transcription: <audio>",
    "Transcribe verbatim: <audio>",
    "Give a verbatim transcription: <audio>",
    "What exact words are spoken? <audio>",
    "Transcribe exactly what is said: <audio>",

    # Content-specific (10)
    "Transcribe the dialogue: <audio>",
    "Transcribe the conversation: <audio>",
    "Transcribe the narration: <audio>",
    "Transcribe the monologue: <audio>",
    "Transcribe the statement: <audio>",
    "Transcribe the announcement: <audio>",
    "Transcribe the comments: <audio>",
    "Transcribe the remarks: <audio>",
    "Transcribe the presentation: <audio>",
    "Transcribe the lecture: <audio>",
]

# Audio Captioning/Description Templates (30 variations)
DESCRIBE_TEMPLATES: List[str] = [
    "Describe: <audio>",
    "Describe the audio: <audio>",
    "Describe this audio: <audio>",
    "Please describe the audio: <audio>",
    "What do you hear? <audio>",
    "What sounds do you hear? <audio>",
    "What is in the audio? <audio>",
    "What is happening in the audio? <audio>",
    "What sounds are present? <audio>",
    "What sounds are in this audio? <audio>",
    "Describe what you hear: <audio>",
    "Describe the sounds: <audio>",
    "Describe the audio content: <audio>",
    "What is the audio about? <audio>",
    "Give a description of the audio: <audio>",
    "Provide a description: <audio>",
    "Characterize the audio: <audio>",
    "Explain what you hear: <audio>",
    "Tell me about the audio: <audio>",
    "What audio events are present? <audio>",
    "Identify the sounds: <audio>",
    "What kind of sounds are these? <audio>",
    "Describe the audio scene: <audio>",
    "What is the soundscape? <audio>",
    "Caption this audio: <audio>",
    "Generate a caption: <audio>",
    "What audio do you perceive? <audio>",
    "Summarize the audio: <audio>",
    "What acoustic events occur? <audio>",
    "Describe the acoustic content: <audio>",
]

# Emotion Recognition Templates (30 variations)
EMOTION_TEMPLATES: List[str] = [
    "Emotion: <audio>",
    "What emotion? <audio>",
    "Identify the emotion: <audio>",
    "What emotion is expressed? <audio>",
    "What emotion do you detect? <audio>",
    "Detect the emotion: <audio>",
    "Recognize the emotion: <audio>",
    "What is the emotional tone? <audio>",
    "What emotion is conveyed? <audio>",
    "What feeling is expressed? <audio>",
    "What is the speaker's emotion? <audio>",
    "How does the speaker feel? <audio>",
    "What emotional state is present? <audio>",
    "Classify the emotion: <audio>",
    "Determine the emotion: <audio>",
    "What sentiment is expressed? <audio>",
    "Identify the emotional content: <audio>",
    "What is the affective state? <audio>",
    "Analyze the emotion: <audio>",
    "What mood is conveyed? <audio>",
    "Detect emotional tone: <audio>",
    "What is the emotional quality? <audio>",
    "Describe the emotion: <audio>",
    "What emotional expression do you hear? <audio>",
    "Identify emotional state: <audio>",
    "What affect is present? <audio>",
    "Recognize emotional content: <audio>",
    "What is the speaker's feeling? <audio>",
    "Categorize the emotion: <audio>",
    "Label the emotion: <audio>",
]

# Template registry
TASK_TEMPLATES: Dict[str, List[str]] = {
    "transcribe": ASR_TEMPLATES,
    "describe": DESCRIBE_TEMPLATES,
    "emotion": EMOTION_TEMPLATES,
}


def get_random_instruction(task: str, seed: int = None) -> str:
    """
    Get a random instruction template for the given task.

    Args:
        task: Task name (transcribe, describe, emotion)
        seed: Optional random seed for reproducibility

    Returns:
        Random instruction template string with <audio> placeholder
    """
    if seed is not None:
        random.seed(seed)

    templates = TASK_TEMPLATES.get(task, ASR_TEMPLATES)
    return random.choice(templates)


def format_instruction(template: str, audio_placeholder: str = "<audio>") -> str:
    """
    Format instruction template (currently just returns template as-is).
    Can be extended for more complex formatting.

    Args:
        template: Instruction template string
        audio_placeholder: Placeholder for audio (default: <audio>)

    Returns:
        Formatted instruction string
    """
    return template
