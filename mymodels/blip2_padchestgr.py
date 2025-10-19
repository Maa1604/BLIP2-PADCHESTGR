from .region_aware_model import RegionBlip2ForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from .model_utility import count_parameters, save_parameter_info


def build_model_and_processor():
    """
    Loads BLIP-2 + LoRA and its processor.
    Returns: (model, processor)
    """
    # Processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

    # Base model (8-bit, sharded)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",   #Salesforce/blip2-opt-2.7b
        device_map="auto",
        dtype=torch.bfloat16, #hace falta para poder entrenar (only Tensors of floating point dtype can require gradients)
        # load_in_8bit=True
    )
    count_parameters(model)


    # --- 0) Freeze the whole model ---
    # (Works for all parameters, including nested modules.)
    model.requires_grad_(False)

    # 1) Unfreeze the requested parts
    # query_tokens is a Parameter, not a module
    model.query_tokens.requires_grad_(True)

    # qformer + language_projection as modules
    model.qformer.requires_grad_(True)
    model.language_projection.requires_grad_(True)


    count_parameters(model)

    save_parameter_info(model, output_file="blip2_parameters.txt")

    return model, processor

def build_grounding_model_and_processor():
    """
    Loads Region-BLIP-2 + LoRA-ready and its processor.
    Returns: (model, processor)
    """
    ckpt = "Salesforce/blip2-opt-2.7b"   #fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray

    # Processor (tokenizer + image processor)
    processor = Blip2Processor.from_pretrained(ckpt)

    # Our subclass can still load the HF checkpoint; extra params (region_token_embed) init randomly
    model = RegionBlip2ForConditionalGeneration.from_pretrained(
        ckpt,
        device_map="auto",
        dtype=torch.bfloat16,  # training-friendly
    )
    count_parameters(model)


    # --- 0) Freeze the whole model ---
    # (Works for all parameters, including nested modules.)
    model.requires_grad_(False)

    # Unfreeze trainable parts
    model.query_tokens.requires_grad_(True)
    model.qformer.requires_grad_(True)
    model.language_projection.requires_grad_(True)
    model.region_token_embed.requires_grad_(True)  # <— add this for W
    count_parameters(model)
    save_parameter_info(model, output_file="blip2_grounded_parameters.txt")

    return model, processor