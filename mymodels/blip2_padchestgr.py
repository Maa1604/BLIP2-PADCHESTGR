from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from .model_utility import count_parameters, save_parameter_info


def build_model_and_processor():
    """
    Loads BLIP-2 + LoRA and its processor.
    Returns: (model, processor)
    """
    # Processor
    processor = Blip2Processor.from_pretrained("fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray")

    # Base model (8-bit, sharded)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray",   #Salesforce/blip2-opt-2.7b
        device_map="auto",
        dtype=torch.bfloat16, #hace falta para poder entrenar (only Tensors of floating point dtype can require gradients)
        # load_in_8bit=True
    )
    count_parameters(model)


    # 1) Unfreeze the requested parts
    # query_tokens is a Parameter, not a module
    model.query_tokens.requires_grad_(True)

    # qformer + language_projection as modules
    model.qformer.requires_grad_(True)
    model.language_projection.requires_grad_(True)


    count_parameters(model)

    save_parameter_info(model, output_file="blip2_parameters.txt")

    exit()

    return model, processor
