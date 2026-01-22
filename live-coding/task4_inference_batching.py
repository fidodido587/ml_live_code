"""
Task 4 – Inference Batching with CPU/GPU Switching

Implement:

    run_inference(model, input_batch, use_gpu=True, max_batch_size=32)

Requirements:
    - If `use_gpu=True` and CUDA is available, move model and data to GPU.
    - Otherwise, run on CPU.
    - Split `input_batch` into smaller chunks of size at most `max_batch_size`.
    - Use `model.eval()` and disable gradient computations during inference.
    - Concatenate all predictions and return them as a single tensor or numpy array.

Assume:
    - `model` is a PyTorch model.
    - `input_batch` is either a torch.Tensor or can be converted to one.
"""

from typing import Union, List

import torch


def run_inference(
    model: torch.nn.Module,
    input_batch: Union[torch.Tensor, "np.ndarray"],
    use_gpu: bool = True,
    max_batch_size: int = 32,
) -> torch.Tensor:
    """
    Run batched inference on CPU or GPU.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    input_batch : torch.Tensor or np.ndarray
        Input data of shape (N, ...).
    use_gpu : bool, optional
        If True and CUDA is available, run on GPU.
    max_batch_size : int, optional
        Maximum batch size for each inference pass.

    Returns
    -------
    outputs : torch.Tensor
        Concatenated model outputs for all inputs.
    """
    # TODO: Implement device selection, batching, and inference loop.
    raise NotImplementedError("Implement run_inference() for Task 4.")


if __name__ == "__main__":
    # Optional: small dummy model + test
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    dummy_model = DummyModel()
    dummy_inputs = torch.randn(100, 10)
    preds = run_inference(dummy_model, dummy_inputs, use_gpu=False, max_batch_size=16)
    print("Predictions shape:", preds.shape)
