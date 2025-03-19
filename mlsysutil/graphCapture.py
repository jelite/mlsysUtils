import torch
import torch.nn as nn
import torch.optim as optim

def warmup(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    loss_fn: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    target: torch.Tensor | None = None,
    train: bool = True
) -> None:
    """Runs a warm-up forward pass (and optional backward pass for training) to initialize CUDA memory.

    Args:
        model (torch.nn.Module): The neural network model.
        input_tensor (torch.Tensor): Input tensor for warm-up.
        loss_fn (torch.nn.Module | None): Loss function (only required for training).
        optimizer (torch.optim.Optimizer | None): Optimizer (only required for training).
        target (torch.Tensor | None): Target tensor for loss computation (only required for training).
        train (bool): Whether to perform training (forward + backward) or just inference (default: True).
    """
    model.train() if train else model.eval()
    warmup_times = 3
    for _ in range(warmup_times):
        with torch.set_grad_enabled(train):
            if train:
                assert loss_fn is not None, "loss_fn must be provided for training"
                assert optimizer is not None, "optimizer must be provided for training"
                assert target is not None, "target must be provided for training"
                output = model(input_tensor)
                optimizer.zero_grad()
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
            # inference
            else: 
                output = model(input_tensor)
    

def get_graph(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    loss_fn: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    target: torch.Tensor | None = None,
    train: bool = True
) -> torch.cuda.CUDAGraph:
    """Captures a CUDA Graph for the given model and input tensor."""
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    warmup(model, input_tensor, train=False)
    
    with torch.cuda.graph(graph):
        static_output = model(input_tensor)

    return graph, static_output