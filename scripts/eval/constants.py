"""Light-weight constants shared by the local and remote eval CLIs.

Kept free of torch/transformers imports so `ta runpod eval` can offer the same
`--assemblyai-model` choices as `ta eval` without paying for the evaluator
stack it never runs locally.
"""

from enum import StrEnum


class AssemblyAIModel(StrEnum):
    """AssemblyAI model options."""

    best = "best"
    universal = "universal"
    universal_3_pro = "universal-3-pro"
    # API name uses dashes for the decimal: "universal-3-5-pro", not "3.5".
    universal_3_5_pro = "universal-3-5-pro"


# Valid `model` values accepted by setup_assemblyai; the enum is the source.
ASSEMBLYAI_MODELS = {m.value for m in AssemblyAIModel}
