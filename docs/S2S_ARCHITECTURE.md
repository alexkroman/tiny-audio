# S2S (Speech-to-Speech) Architecture

## How It Works

1. **Audio In** — Capture speech from microphone or file

1. **Understand Speech** — Audio encoder converts sound waves into features the model can process

1. **Bridge to Language** — Projector translates audio features into the language model's format

1. **Think** — Language model processes the input and generates a response (as hidden states)

1. **Prepare for Speech** — Flow network converts the response into audio latents (a compact audio representation)

1. **Audio Out** — Mimi decoder turns latents back into audible speech

## Training

Only the flow network is trained. Everything else (encoder, projector, language model, decoder) stays frozen from previous training stages.
