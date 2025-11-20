import torch
import numpy as np
from datasets import load_dataset
from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64

def main():
    # --- 1. Load a single audio sample ---
    print("Loading audio sample...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = dataset[0]["audio"]
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]
    reference_text = dataset[0]["text"]
    print(f"✓ Audio loaded. Duration: {len(waveform)/sampling_rate:.2f}s, Rate: {sampling_rate} Hz")
    print(f"  Reference text: {reference_text}")

    # --- 2. Load the full ASR model ---
    print("\nLoading ASR model (this may take a moment)...")
    config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
    model = ASRModel(config)

    try:
        encoder_device = next(model.encoder.parameters()).device
    except StopIteration:
        encoder_device = "cpu"

    model.projector = model.projector.to(encoder_device)
    model.eval()
    print(f"✓ Model loaded to '{encoder_device}' device.")

    # --- 3. Process data ---
    print("\nProcessing audio...")
    
    # Get Spectrogram
    features = model.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = features.input_features
    spectrogram = input_features.squeeze(0).cpu().numpy()
    print(f"  Spectrogram shape: {spectrogram.shape}")
    
    # Get Encoder Output
    with torch.no_grad():
        encoder_dtype = next(model.encoder.parameters()).dtype
        input_features = input_features.to(device=encoder_device, dtype=encoder_dtype)
        encoder_output = model.encoder(input_features).last_hidden_state
    
    # Get Projector Output
    with torch.no_grad():
        projector_output = model.projector(encoder_output)

    # Decode projector output to show "text-like" nature
    # Find nearest text embeddings for each time step
    with torch.no_grad():
        # Get the text embedding matrix from the decoder
        text_embeddings = model.decoder.get_input_embeddings().weight  # [vocab_size, hidden_dim]

        # Move to same device as projector output
        device = projector_output.device
        text_embeddings = text_embeddings.to(device)

        # Compute similarity between projector output and all text embeddings
        # projector_output: [1, seq_len, hidden_dim]
        # text_embeddings: [vocab_size, hidden_dim]
        projector_flat = projector_output.squeeze(0)  # [seq_len, hidden_dim]

        # Normalize for cosine similarity
        proj_norm = projector_flat / projector_flat.norm(dim=-1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # Get top token for each time step
        similarities = torch.matmul(proj_norm, text_norm.T)  # [seq_len, vocab_size]
        top_tokens = similarities.argmax(dim=-1)  # [seq_len]

        # Decode to text
        nearest_tokens = model.tokenizer.batch_decode(top_tokens.cpu().unsqueeze(1), skip_special_tokens=False)
        nearest_tokens_text = " ".join([t.strip() for t in nearest_tokens])  # All tokens

    print("✓ Data processing complete.")
    print(f"  Nearest text tokens (all {len(nearest_tokens)} tokens):")
    print(f"  {nearest_tokens_text}")

    # --- 4. Prepare data for visualization ---
    print("\nPreparing data for visualization...")
    # Downsample waveform for faster plotting
    waveform_downsampled = waveform[::10].tolist()
    audio_duration = len(waveform) / sampling_rate  # Calculate duration from original waveform

    # Use full embedding space for visualization
    encoder_viz_data = encoder_output.squeeze(0).cpu().float().numpy()
    projector_viz_data = projector_output.squeeze(0).cpu().float().numpy()

    data_payload = {
        "reference_text": reference_text,
        "waveform": waveform_downsampled,
        "sampling_rate": sampling_rate,
        "audio_duration": audio_duration,
        "spectrogram": {
            "values": spectrogram.tolist(),
            "width": spectrogram.shape[1],
            "height": spectrogram.shape[0]
        },
        "encoder_output": {
            "values": encoder_viz_data.T.tolist(),
            "width": encoder_viz_data.shape[0],
            "height": encoder_viz_data.shape[1],
            "stats": get_stats(encoder_output, "Encoder")
        },
        "projector_output": {
            "values": projector_viz_data.T.tolist(),
            "width": projector_viz_data.shape[0],
            "height": projector_viz_data.shape[1],
            "stats": get_stats(projector_output, "Projector"),
            "nearest_tokens": nearest_tokens_text
        }
    }
    
    # --- 5. HTML Generation ---
    print("\nGenerating HTML report...")
    html_content = generate_observable_html(data_payload)

    with open("data_trace.html", "w") as f:
        f.write(html_content)
    print(f"\n✓ Saved HTML report to 'data_trace.html'")


def get_stats(tensor, name):
    return {
        "name": name,
        "shape": str(tuple(tensor.shape)),
        "mean": f"{tensor.mean():.4f}",
        "std": f"{tensor.std():.4f}",
        "min": f"{tensor.min():.4f}",
        "max": f"{tensor.max():.4f}"
    }

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def create_pipeline_summary(data):
    """Create a summary diagram showing the transformation pipeline."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    # Calculate dimensions
    spec_shape = (data['spectrogram']['height'], data['spectrogram']['width'])
    enc_shape = (data['encoder_output']['height'], data['encoder_output']['width'])
    proj_shape = (data['projector_output']['height'], data['projector_output']['width'])

    # Draw boxes
    boxes = [
        (0.05, 0.3, 0.15, 0.4, f"Waveform\n{len(data['waveform'])}×1", '#e8f4f8'),
        (0.25, 0.3, 0.15, 0.4, f"Spectrogram\n{spec_shape[0]}×{spec_shape[1]}", '#d4e9f7'),
        (0.45, 0.3, 0.15, 0.4, f"Encoder\n{enc_shape[1]}×{enc_shape[0]}", '#b8daf0'),
        (0.65, 0.3, 0.15, 0.4, f"Projector\n{proj_shape[1]}×{proj_shape[0]}", '#9cc9e8'),
        (0.85, 0.3, 0.15, 0.4, "Decoder\n(LLM)", '#7fb3d5')
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#333', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=9, fontweight='bold')

    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#333')
    arrows = [
        (0.20, 0.5, 0.05, 0),
        (0.40, 0.5, 0.05, 0),
        (0.60, 0.5, 0.05, 0),
        (0.80, 0.5, 0.05, 0)
    ]

    for x, y, dx, dy in arrows:
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y), arrowprops=arrow_props)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_title('Audio-to-Text Pipeline: Dimensional Transformations',
                 fontsize=12, fontweight='bold', pad=20)

    return fig_to_base64(fig)

def generate_observable_html(data):
    """Generate HTML report with matplotlib visualizations."""

    # Create visualizations
    print("Generating visualizations...")

    # 0. Pipeline summary
    pipeline_summary_img = create_pipeline_summary(data)

    # 1. Waveform plot
    fig, ax = plt.subplots(figsize=(12, 3))
    time_axis = np.arange(len(data['waveform'])) * 10 / data['sampling_rate']
    ax.plot(time_axis, data['waveform'], color='royalblue', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('1. Raw Audio Waveform')
    ax.grid(True, alpha=0.3)
    waveform_img = fig_to_base64(fig)

    # 2. Spectrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    spec_data = np.array(data['spectrogram']['values'])
    # Spectrogram is [freq_bins, time_steps] = [128, 3000]
    im = ax.imshow(spec_data, aspect='auto', origin='lower', cmap='viridis')

    # Manually set x-axis to show time in seconds
    num_time_steps = spec_data.shape[1]
    audio_duration = data['audio_duration']
    # Set 5-6 tick marks across the time axis
    num_ticks = 6
    tick_positions = np.linspace(0, num_time_steps - 1, num_ticks)
    tick_labels = [f"{audio_duration * (pos / (num_time_steps - 1)):.2f}" for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency Bin')
    ax.set_title('2. Log-Mel Spectrogram (Input to Encoder)')
    plt.colorbar(im, ax=ax, label='dB')
    spectrogram_img = fig_to_base64(fig)

    # 3. Encoder output (show most active dimensions)
    fig, ax = plt.subplots(figsize=(12, 6))
    encoder_data = np.array(data['encoder_output']['values'])
    # Select the top 64 most active dimensions based on variance
    variances = np.var(encoder_data, axis=1)
    top_dims = np.argsort(variances)[-64:]  # Get indices of top 64 most varying dimensions
    encoder_data_subset = encoder_data[top_dims, :]
    # Use percentile-based color scaling for better contrast
    vmin, vmax = np.percentile(encoder_data_subset, [1, 99])
    im = ax.imshow(encoder_data_subset, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(f"Embedding Dimension (64 most active of {data['encoder_output']['height']} dims)")
    ax.set_title('3. Encoder Output (Audio Embeddings - Most Active Dimensions)')
    plt.colorbar(im, ax=ax, label='Activation')
    encoder_img = fig_to_base64(fig)

    # 4. Projector output (show most active dimensions)
    fig, ax = plt.subplots(figsize=(12, 6))
    projector_data = np.array(data['projector_output']['values'])
    # Select the top 64 most active dimensions based on variance
    variances = np.var(projector_data, axis=1)
    top_dims = np.argsort(variances)[-64:]  # Get indices of top 64 most varying dimensions
    projector_data_subset = projector_data[top_dims, :]
    # Use percentile-based color scaling for better contrast
    vmin, vmax = np.percentile(projector_data_subset, [1, 99])
    im = ax.imshow(projector_data_subset, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(f"Embedding Dimension (64 most active of {data['projector_output']['height']} dims)")
    ax.set_title('4. Projector Output (Text-like Embeddings - Most Active Dimensions)')
    plt.colorbar(im, ax=ax, label='Activation')
    projector_img = fig_to_base64(fig)

    # 5. SwiGLU architecture visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Show the SwiGLU formula flow
    ax = axes[0, 0]
    ax.axis('off')
    ax.text(0.5, 0.95, 'SwiGLU Architecture', ha='center', va='top',
            fontsize=14, fontweight='bold')

    # Draw boxes for each step
    boxes = [
        (0.5, 0.85, 'Input Vector\n[2560 dims]', '#e8f4f8'),
        (0.5, 0.70, 'RMSNorm (pre)', '#d4e9f7'),
        (0.25, 0.55, 'W1 × x\n↓\nSiLU(x)', '#b8daf0'),
        (0.75, 0.55, 'W2 × x\n(no activation)', '#b8daf0'),
        (0.5, 0.40, 'Gate * Value\n(element-wise)', '#9cc9e8'),
        (0.5, 0.25, 'W3 × x', '#7fb3d5'),
        (0.5, 0.10, 'RMSNorm (post)\n↓\nOutput [1536 dims]', '#6ba3c9'),
    ]

    for x, y, text, color in boxes:
        width = 0.35 if 'Gate' not in text and 'W1' not in text and 'W2' not in text else 0.25
        rect = plt.Rectangle((x - width/2, y - 0.05), width, 0.08,
                            facecolor=color, edgecolor='#333', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#333')
    ax.annotate('', xy=(0.5, 0.77), xytext=(0.5, 0.82), arrowprops=arrow_props)
    ax.annotate('', xy=(0.25, 0.62), xytext=(0.5, 0.67), arrowprops=arrow_props)
    ax.annotate('', xy=(0.75, 0.62), xytext=(0.5, 0.67), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.47), xytext=(0.25, 0.52), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.47), xytext=(0.75, 0.52), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.32), xytext=(0.5, 0.37), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.17), xytext=(0.5, 0.22), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Top right: SiLU activation function
    ax = axes[0, 1]
    x_vals = np.linspace(-5, 5, 1000)
    silu_vals = x_vals / (1 + np.exp(-x_vals))
    ax.plot(x_vals, silu_vals, linewidth=2, color='#1f77b4')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Input Value', fontsize=10)
    ax.set_ylabel('Output Value', fontsize=10)
    ax.set_title('SiLU Activation Function\ny = x / (1 + e^-x)', fontsize=11, fontweight='bold')
    ax.text(0.05, 0.95, 'Non-linear, smooth\nAllows gradients to flow',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom left: Gating mechanism visualization
    ax = axes[1, 0]
    # Simulate gate and value vectors
    np.random.seed(42)
    n_dims = 20
    gate = np.random.uniform(0, 1, n_dims)  # After SiLU, values are mostly positive
    value = np.random.randn(n_dims)
    gated = gate * value

    x_pos = np.arange(n_dims)
    width = 0.25
    ax.barh(x_pos - width, gate, width, label='Gate (SiLU)', color='#1f77b4', alpha=0.7)
    ax.barh(x_pos, value, width, label='Value (linear)', color='#ff7f0e', alpha=0.7)
    ax.barh(x_pos + width, gated, width, label='Gate × Value', color='#2ca02c', alpha=0.7)
    ax.set_ylabel('Dimension Index', fontsize=10)
    ax.set_xlabel('Activation Value', fontsize=10)
    ax.set_title('Gating Mechanism (20 dims shown)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Bottom right: Why non-linearity matters
    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.5, 0.95, 'Why SwiGLU vs Simple Linear?', ha='center', va='top',
            fontsize=12, fontweight='bold')

    explanation_text = """
Simple Linear: output = W × input
• Direct, proportional mapping
• Can't model complex relationships
• Example: "always boost dim 5 by 2x"

SwiGLU: output = W3(SiLU(W1×x) ⊙ W2×x)
• Conditional, context-aware mapping
• Models complex relationships
• Example: "boost dim 5 by 2x IF dims 3
  and 7 are high AND dim 12 is low"

The gating mechanism (⊙ = multiply) allows
selective activation - like a learned "if-then"
rule for each dimension.

This is crucial because audio→text isn't a
simple linear transformation. The relationship
between "hearing 'k' sound" and "writing 'k'
token" depends on context!
"""
    ax.text(0.05, 0.80, explanation_text, ha='left', va='top', fontsize=8.5,
            family='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    swiglu_img = fig_to_base64(fig)

    # Generate HTML
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASR Data Trace</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 2em;
            background-color: #f0f2f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }}
        .section {{
            margin: 30px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        code {{
            background-color: #e8eaed;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>How Speech Becomes Text: An Interactive Visualization</h1>
        <p>Watch how your spoken words transform into written text, step by step.</p>

        <div class="section">
            <h2>The Big Picture</h2>
            <p>Converting speech to text is like translating between two completely different languages. Sound waves are continuous and flowing, while text is discrete symbols. This model learns to bridge that gap.</p>
            <img src="{pipeline_summary_img}" alt="Pipeline Summary">
        </div>

        <div class="section">
            <h2>Input: The Audio Sample</h2>
            <p><strong>Reference Text:</strong> <em>"{data['reference_text']}"</em></p>
            <p>This is what the speaker actually said. Our goal is to recover this text from the raw audio signal alone.</p>
        </div>

        <div class="section">
            <h2>Step 1: Sound Waves</h2>
            <p>Speech starts as vibrations in the air. A microphone captures these vibrations {data['sampling_rate']:,} times per second, turning sound into numbers.</p>
            <img src="{waveform_img}" alt="Waveform">
            <p><em>The ups and downs show how loud the sound is at each moment. Bigger waves = louder sounds.</em></p>
        </div>

        <div class="section">
            <h2>Step 2: Frequency Analysis (Spectrogram)</h2>
            <p>Like a musical score shows different notes over time, a spectrogram shows different frequencies in the speech. Think of it as converting sound into a heat map.</p>
            <img src="{spectrogram_img}" alt="Spectrogram">
            <p><em>Time flows left to right. Bottom = low sounds (like bass), Top = high sounds (like whistles). Bright = loud, Dark = quiet. The purple area on the right is just padding - ignore it.</em></p>

            <h3>What we're seeing:</h3>
            <ul>
                <li><strong>Bottom rows:</strong> Deep voice tones</li>
                <li><strong>Middle rows:</strong> Vowel sounds (a, e, i, o, u)</li>
                <li><strong>Top rows:</strong> Consonants like 's' and 'f'</li>
            </ul>
        </div>

        <div class="section">
            <h2>Step 3: Understanding Speech Sounds (Encoder)</h2>
            <p>The AI "listens" to the spectrogram and identifies speech patterns. It recognizes things like individual sounds, speaking pace, and voice characteristics.</p>
            <img src="{encoder_img}" alt="Encoder Output">
            <p><em>This is what the AI "understands" about the audio. Each column is a moment in time. Brighter colors mean the AI detected something important at that moment.</em></p>

            <h3>What it's detecting:</h3>
            <ul>
                <li>Individual speech sounds (like "t", "s", "ah")</li>
                <li>Whether sounds are voiced or whispered</li>
                <li>Pitch and rhythm patterns</li>
                <li>Speaker characteristics (accent, gender, age)</li>
            </ul>
            <p><strong>Important:</strong> At this stage, "to," "too," and "two" all look the same because they sound the same! The AI knows the sound but not yet the spelling.</p>
        </div>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Shape</td><td><code>{data['encoder_output']['stats']['shape']}</code></td></tr>
                <tr><td>Mean</td><td><code>{data['encoder_output']['stats']['mean']}</code></td></tr>
                <tr><td>Std Dev</td><td><code>{data['encoder_output']['stats']['std']}</code></td></tr>
                <tr><td>Min</td><td><code>{data['encoder_output']['stats']['min']}</code></td></tr>
                <tr><td>Max</td><td><code>{data['encoder_output']['stats']['max']}</code></td></tr>
            </table>
        </div>

        <div class="section">
            <h2>Step 4: Translating to Text (Projector)</h2>
            <p>The projector's job: convert "what was heard" into "how to write it." This is where sound becomes text-ready.</p>
            <img src="{projector_img}" alt="Projector Output">
            <p><em>Now the AI is thinking in text, not sound. It's figuring out spelling, punctuation, and capitalization.</em></p>

            <h3>The key trick:</h3>
            <p>The projector solves problems that sound alone can't answer:</p>
            <ul>
                <li><strong>"to" vs "too" vs "two"</strong> - Same sound, different spelling based on meaning</li>
                <li><strong>"their" vs "there" vs "they're"</strong> - Context determines which one</li>
                <li><strong>Question marks</strong> - Rising tone at the end → add "?"</li>
                <li><strong>Capitalization</strong> - "apple" (fruit) vs "Apple" (company)</li>
            </ul>
        </div>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Shape</td><td><code>{data['projector_output']['stats']['shape']}</code></td></tr>
                <tr><td>Mean</td><td><code>{data['projector_output']['stats']['mean']}</code></td></tr>
                <tr><td>Std Dev</td><td><code>{data['projector_output']['stats']['std']}</code></td></tr>
                <tr><td>Min</td><td><code>{data['projector_output']['stats']['min']}</code></td></tr>
                <tr><td>Max</td><td><code>{data['projector_output']['stats']['max']}</code></td></tr>
            </table>

            <h3>Checking: Does it Look Like Text?</h3>
            <p><strong>What was actually said:</strong> <em>"{data['reference_text']}"</em></p>
            <p>If we check what vocabulary words are closest to each moment, we get this jumble:</p>
            <p style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; font-family: 'Courier New', monospace; max-height: 150px; overflow-y: scroll; font-size: 11px;">
                {data['projector_output']['nearest_tokens']}
            </p>
            <p><em>This isn't the final transcription! But it proves the projector is working—the AI is now thinking in text-like patterns. The language model will clean this up into proper sentences.</em></p>
        </div>

        <div class="section">
            <h2>How The Projector Works: Smart Gating</h2>
            <p>The projector uses a clever trick called SwiGLU - think of it like a smart gate that decides what information passes through.</p>
            <img src="{swiglu_img}" alt="SwiGLU Architecture">

            <h3>Simple Analogy:</h3>
            <p>Imagine you're hearing the "s" sound. The projector asks: "Is there voice vibration?"</p>
            <ul>
                <li>No vibration → write "s" (like in "snake")</li>
                <li>Yes vibration → write "z" (like in "buzz")</li>
            </ul>
            <p>It makes <strong>conditional decisions</strong> based on context, not just blind copying.</p>

            <h3>In The Visualization:</h3>
            <ul>
                <li><strong>Top Left:</strong> The flowchart shows how information splits into two paths and combines intelligently</li>
                <li><strong>Bottom Left:</strong> Blue bars act as gates controlling orange bars, producing green output</li>
            </ul>
            <p><em>The key insight: Simple math can't decide "to" vs "too" vs "two" - you need context-aware gating.</em></p>
        </div>

        <div class="section">
            <h2>The Big Picture</h2>
            <p>You've just watched speech recognition happen in slow motion:</p>
            <ol>
                <li><strong>Sound waves</strong> → captured as numbers</li>
                <li><strong>Spectrogram</strong> → frequencies over time</li>
                <li><strong>Encoder</strong> → understands "what was said"</li>
                <li><strong>Projector</strong> → translates to "how to write it"</li>
            </ol>
            <p>The magic? We only had to train the tiny projector (step 4). The encoder and language model were already trained on massive datasets. This makes building powerful speech recognition affordable for everyone.</p>
        </div>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    main()
