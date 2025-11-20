import torch
import numpy as np
from datasets import load_dataset
from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel
import json
import os

def main():
    # --- 1. Load a single audio sample ---
    print("Loading audio sample...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    audio_sample = dataset[0]["audio"]
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]
    print(f"✓ Audio loaded. Duration: {len(waveform)/sampling_rate:.2f}s, Rate: {sampling_rate} Hz")

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
    
    # Get Encoder Output
    with torch.no_grad():
        encoder_dtype = next(model.encoder.parameters()).dtype
        input_features = input_features.to(device=encoder_device, dtype=encoder_dtype)
        encoder_output = model.encoder(input_features).last_hidden_state
    
    # Get Projector Output
    with torch.no_grad():
        projector_output = model.projector(encoder_output)
    
    print("✓ Data processing complete.")

    # --- 4. Prepare data for Observable ---
    print("\nPreparing data for visualization...")
    # Downsample waveform for faster plotting
    waveform_downsampled = waveform[::10].tolist()

    # For performance, only visualize a subset of embedding dimensions
    num_dims_to_visualize = 64
    encoder_viz_data = encoder_output.squeeze(0)[:, :num_dims_to_visualize].cpu().float().numpy()
    projector_viz_data = projector_output.squeeze(0)[:, :num_dims_to_visualize].cpu().float().numpy()

    data_payload = {
        "waveform": waveform_downsampled,
        "sampling_rate": sampling_rate,
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
            "stats": get_stats(projector_output, "Projector")
        }
    }
    
    # --- 5. HTML Generation ---
    print("\nGenerating Observable JS HTML report...")
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

def generate_observable_html(data):
    # This function now embeds the data and the Observable JS rendering code into a single HTML file.
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASR Data Trace</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@observablehq/inspector@5/dist/inspector.css">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif; margin: 2em; background-color: #f0f2f5; }}
        #notebook {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        code {{ background-color: #e8eaed; padding: 2px 6px; border-radius: 4px; }}
    </style>
</head>
<body>
<div id="notebook"></div>

<script type="module">
    import {{Runtime, Inspector}} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/dist/runtime.js";
    import define from "https://cdn.jsdelivr.net/npm/@observablehq/stdlib@5/dist/stdlib.js";

    const notebook = {{
        "doc": [
            {{
                "id": "intro",
                "value": "md`# The Journey of an Audio Signal\nThis report visualizes how a raw audio waveform is transformed step-by-step into a format that a Large Language Model can understand.`"
            }},
            {{
                "id": "data",
                "value": `{json.dumps(data)}`
            }},
            {{
                "id": "waveform_title",
                "value": "md`## 1. Raw Audio Waveform`"
            }},
            {{
                "id": "waveform_chart",
                "value": `
Plot.plot({{
    marks: [
        Plot.lineY(data.waveform, {{ 
            x: (d, i) => i * 10 / data.sampling_rate, // Downsampled by 10x
            y: d,
            stroke: "royalblue"
        }}),
        Plot.ruleY([0])
    ],
    x: {{ label: "Time (s)" }},
    y: {{ label: "Amplitude", grid: true }}
}})`
            }},
            {{
                "id": "spectrogram_title",
                "value": "md`## 2. Log-Mel Spectrogram (Input to Encoder)`"
            }},
            {{
                "id": "spectrogram_chart",
                "value": `
Plot.plot({{
    marks: [
        Plot.raster(data.spectrogram.values, {{ 
            x: (d, i) => i % data.spectrogram.width,
            y: (d, i) => Math.floor(i / data.spectrogram.width),
            fill: d => d,
            imageRendering: "pixelated"
        }})
    ],
    color: {{ scheme: "viridis", label: "dB" }},
    y: {{ label: "Mel Frequency Bin" }},
    x: {{ label: "Time Frame" }}
}})`
            }},
            {{
                "id": "encoder_title",
                "value": `md\
## 3. Encoder Output (Audio Embeddings)
|
Property | Value |
|---|---|
| Shape | \
`
${{data.encoder_output.stats.shape}}\
` |
| Mean  | \
`
${{data.encoder_output.stats.mean}}\
` |
| Std Dev| \
`
${{data.encoder_output.stats.std}}\
` |
| Min   | \
`
${{data.encoder_output.stats.min}}\
` |
| Max   | \
`
${{data.encoder_output.stats.max}}\
` |



`
            }},
            {{
                "id": "encoder_chart",
                "value": `
Plot.plot({{
    marks: [
        Plot.raster(data.encoder_output.values.flat(), {{ 
            x: (d, i) => i % data.encoder_output.width,
            y: (d, i) => Math.floor(i / data.encoder_output.width),
            fill: d => d
        }})
    ],
    color: {{ scheme: "viridis", label: "Activation" }},
    y: {{ label: "Embedding Dimension (first ${{data.encoder_output.height}})" }},
    x: {{ label: "Time Steps" }}
}})`
            }},
            {{
                "id": "projector_title",
                "value": `md\
## 4. Projector Output (Text-like Embeddings)
|
Property | Value |
|---|---|
| Shape | \
`
${{data.projector_output.stats.shape}}\
` |
| Mean  | \
`
${{data.projector_output.stats.mean}}\
` |
| Std Dev| \
`
${{data.projector_output.stats.std}}\
` |
| Min   | \
`
${{data.projector_output.stats.min}}\
` |
| Max   | \
`
${{data.projector_output.stats.max}}\
` |



`
            }},
            {{
                "id": "projector_chart",
                "value": `
Plot.plot({{
    marks: [
        Plot.raster(data.projector_output.values.flat(), {{ 
            x: (d, i) => i % data.projector_output.width,
            y: (d, i) => Math.floor(i / data.projector_output.width),
            fill: d => d
        }})
    ],
    color: {{ scheme: "viridis", label: "Activation" }},
    y: {{ label: "Embedding Dimension (first ${{data.projector_output.height}})" }},
    x: {{ label: "Time Steps" }}
}})`
            }}
        ]
    }};

    const main = new Runtime().module(notebook, name => {{
        const div = document.createElement("div");
        document.getElementById("notebook").append(div);
        return new Inspector(div);
    }});

</script>
</body>
</html>
""";

if __name__ == "__main__":
    main()
