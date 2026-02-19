from torch import nn


class Perceptron(nn.Module):
    """Baseline: one linear layer -> logit."""

    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, features):
        return self.linear(features)


class MLP(nn.Module):
    """Main model: 2-layer MLP -> logit."""

    def __init__(self, num_features, hidden_sizes, dropout):
        super().__init__()
        hidden_1, hidden_2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, features):
        return self.net(features)
