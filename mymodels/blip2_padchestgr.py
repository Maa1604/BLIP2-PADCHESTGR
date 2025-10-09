# mymodels/blip2_lora.py

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel


def build_model_and_processor():
    """
    Loads BLIP-2 + LoRA and its processor.
    Returns: (model, processor)
    """
    # Processor
    processor = AutoProcessor.from_pretrained("fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray")

    # Base model (8-bit, sharded)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray",
        device_map="auto",
        load_in_8bit=True
    )
    #print(model.peft_config) #ver el lora anterior
    output_file = "blip2_parameters.txt"

    with open(output_file, "w") as f:
        for name, param in model.named_parameters():
            num_params = param.numel()
            f.write(f"{name:80} | shape={tuple(param.shape)} | params={num_params:,} | requires_grad={param.requires_grad}\n")

    print(f"✅ Parameter list saved to {output_file}")
    
    #Apply LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.01,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    exit()

    return model, processor
