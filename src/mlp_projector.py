import torch.nn as nn


class MLPAudioProjector(nn.Module):
    """2-layer MLP projector with Qwen-style 2x temporal downsampling."""

    def __init__(self, config):
        super().__init__()

        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 2048)

        self.downsample = nn.Conv1d(
            encoder_dim, encoder_dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.linear_1 = nn.Linear(encoder_dim, llm_dim, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_dim, llm_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        Returns: [Batch, Seq_Len // 2, llm_dim]
        """
        # Conv1d expects [Batch, Channels, Seq_Len]
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)

        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)
