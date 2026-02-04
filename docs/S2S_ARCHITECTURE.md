# S2S (Speech-to-Speech) Architecture

## Step 1: Audio Input

Raw audio waveform is captured from a microphone or loaded from a file. The audio is typically sampled at 16kHz.

## Step 2: Audio Encoding

The audio waveform is passed through a frozen GLM-ASR encoder (or Whisper encoder), which converts the raw audio into a sequence of dense audio feature vectors representing the speech content.

## Step 3: Projector Mapping

The audio features are passed through a trained MLP projector that bridges the audio encoder's representation space to the language model's embedding space. The projector uses frame stacking (stride of 4) to downsample the sequence and align audio frames to text token positions.

## Step 4: Language Model Processing

The projected audio embeddings are fed into a frozen SmolLM3-3B language model. The LLM processes the audio representations and produces hidden states that encode both the semantic understanding of the speech and the context for generating a response.

## Step 5: Depformer Token Generation

The LLM hidden states are passed to a Moshi-style Depformer (Depth Transformer). For each time step, the Depformer autoregressively generates 8 codebook tokens, where each subsequent codebook is conditioned on the previous codebook's token output.

## Step 6: Codec Decoding

The 8 codebook token sequences are passed to a Mimi neural codec decoder, which reconstructs the audio waveform from the discrete token representation.

## Step 7: Audio Output

The reconstructed audio waveform is played through speakers or saved to a file, completing the speech-to-speech pipeline.
