"""
Task 3 – Debug a PyTorch Training Snippet

The script below is intentionally broken. Your job is to:

1. Identify the bugs.
2. Fix them.
3. Explain each fix.

Do NOT rewrite the whole script from scratch; focus on making minimal, correct fixes.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BUG: returns the layer object instead of applying it to x.
        # Candidate should fix this.
        return self.lin  # noqa: E501


def main() -> None:
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X = torch.randn(64, 10)
    y = torch.randn(64, 1)

    for i in range(100):
        # BUGS: missing zero_grad, wrong forward, etc.
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i} - Loss: {loss.item()}")


if __name__ == "__main__":
    main()
