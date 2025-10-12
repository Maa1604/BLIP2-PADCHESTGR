def count_parameters(model):
    """
    Prints the total and trainable parameters of a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_trainable = (trainable_params / total_params) * 100 if total_params > 0 else 0


    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({percent_trainable:.2f}%)")
