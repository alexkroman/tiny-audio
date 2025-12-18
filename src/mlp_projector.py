import torch.nn as nn


class MLPAudioProjector(nn.Module):
    """Simple 2-layer MLP projector."""

    def __init__(self, config):
        super().__init__()

        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 4096)

        self.linear_1 = nn.Linear(encoder_dim, llm_dim, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_dim, llm_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        """
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x
