#!/usr/bin/env python3
"""Test multi-task inference with different prompts."""

def test_multitask_inference():
    """Example of how to use the model with different tasks."""

    from src.asr_pipeline import ASRPipeline
    from src.asr_modeling import ASRModel

    # Load model (use your trained model path)
    model_path = "outputs/multi_task_all_test_model"  # Update with actual path
    # model = ASRModel.from_pretrained(model_path)

    # Create pipeline
    # pipeline = ASRPipeline(model)

    # Example usage for different tasks
    print("Multi-task inference examples:")
    print("-" * 50)

    # 1. Transcription task (default)
    print("1. Transcription task:")
    print("   pipeline(audio_file)")
    print("   # or explicitly:")
    print("   pipeline(audio_file, task='transcribe')")
    print("   # Uses prompt: 'Transcribe: <audio>'")
    print()

    # 2. Continuation task
    print("2. Continuation task:")
    print("   pipeline(audio_file, task='continue')")
    print("   # Uses prompt: 'Continue: <audio>'")
    print("   # Predicts the next turn in a dialogue")
    print()

    # 3. Description task
    print("3. Description task:")
    print("   pipeline(audio_file, task='describe')")
    print("   # Uses prompt: 'Describe: <audio>'")
    print("   # Generates a description of sounds in the audio")
    print()

    # You can also use the model directly
    print("Direct model usage:")
    print("-" * 50)
    print("""
    # Process audio
    input_values = model.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    # Generate with specific task
    output = model.generate(
        input_values,
        task="describe",  # or "transcribe", "continue"
        max_new_tokens=200
    )

    # Decode output
    text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    """)

if __name__ == "__main__":
    test_multitask_inference()