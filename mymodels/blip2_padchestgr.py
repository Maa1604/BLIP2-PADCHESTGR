from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch
from .model_utility import count_parameters


def build_model_and_processor():
    """
    Loads BLIP-2 + LoRA and its processor.
    Returns: (model, processor)
    """
    # Processor
    processor = AutoProcessor.from_pretrained("fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray") #TODO: CHECK THIS

    # Base model (8-bit, sharded)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray",   #Salesforce/blip2-opt-2.7b
        device_map="auto",
        dtype=torch.bfloat16, #hace falta para poder entrenar (only Tensors of floating point dtype can require gradients)
        # load_in_8bit=True
    )
    #print(model.peft_config) #ver el lora anterior
    # output_file = "blip2_parameters.txt"

    # with open(output_file, "w") as f:
    #     for name, param in model.named_parameters():
    #         num_params = param.numel()
    #         f.write(f"{name:80} | shape={tuple(param.shape)} | params={num_params:,} | requires_grad={param.requires_grad}\n")

    # print(f"✅ Parameter list saved to {output_file}")
    
    #Apply LoRA
    # lora_cfg = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.01,
    #     bias="none",
    #     target_modules=["qformer.query_tokens", "language_model"]
    # )
    # model = get_peft_model(model, lora_cfg)
    # Count total parameters
    count_parameters(model)


    # 1) Unfreeze the requested parts
    # query_tokens is a Parameter, not a module
    model.query_tokens.requires_grad_(True)

    # qformer + language_projection as modules
    model.qformer.requires_grad_(True)
    model.language_projection.requires_grad_(True)


    count_parameters(model)

    output_file = "blip2_parameters.txt"

    with open(output_file, "w") as f:
        for name, param in model.named_parameters():
            num_params = param.numel()
            f.write(f"{name:80} | shape={tuple(param.shape)} | params={num_params:,} | requires_grad={param.requires_grad}\n")


    exit()

    return model, processor
