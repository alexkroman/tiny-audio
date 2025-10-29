from transformers import pipeline

# Load the pre-trained Tiny Audio model
print("Loading model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True
)

print("✓ Model loaded!")

# Transcribe an audio file
# Replace with your own audio file path
audio_path = "path/to/your/audio.wav"

print(f"Transcribing {audio_path}...")
result = pipe(audio_path)

print("\nTranscription:")
print(result["text"])
