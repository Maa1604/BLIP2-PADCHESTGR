from .region_aware_model import RegionBlip2ForConditionalGeneration
from transformers import Blip2Processor
import torch
from .model_utility import count_parameters, save_parameter_info

def build_model_and_processor():
    """
    Loads Region-BLIP-2 + LoRA-ready and its processor.
    Returns: (model, processor)
    """
    ckpt = "fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray"

    # Processor (tokenizer + image processor)
    processor = Blip2Processor.from_pretrained(ckpt)

    # Our subclass can still load the HF checkpoint; extra params (region_token_embed) init randomly
    model = RegionBlip2ForConditionalGeneration.from_pretrained(
        ckpt,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # training-friendly
    )
    count_parameters(model)


    # Unfreeze trainable parts
    model.query_tokens.requires_grad_(True)
    model.qformer.requires_grad_(True)
    model.language_projection.requires_grad_(True)
    model.region_token_embed.requires_grad_(True)  # <— add this for W
    count_parameters(model)
    save_parameter_info(model, output_file="blip2_grounded_parameters.txt")
    exit()

    return model, processor
