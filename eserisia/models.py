import torch.nn as nn


class TransformerEvolved(nn.Module):
    """Minimal stub of a transformer-like model."""

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        evolution_enabled: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.evolution_enabled = evolution_enabled
        # Simple linear layer as placeholder
        self.dummy = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.dummy(x)


class LiquidNeuralNetwork(nn.Module):
    """Minimal stub of a liquid neural network."""

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 1024,
        adaptation_rate: float = 0.01,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adaptation_rate = adaptation_rate
        self.dummy = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return self.dummy(x)
