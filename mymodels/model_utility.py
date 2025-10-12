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

def save_parameter_info(model, output_file="blip2_parameters.txt"):
        """
        Saves detailed parameter information to a text file and exits the program.
        """
        with open(output_file, "w") as f:
            for name, param in model.named_parameters():
                num_params = param.numel()
                f.write(
                    f"{name:80} | shape={tuple(param.shape)} | "
                    f"params={num_params:,} | requires_grad={param.requires_grad}\n"
                )

        print(f"Parameter information saved to {output_file}. Exiting now.")
