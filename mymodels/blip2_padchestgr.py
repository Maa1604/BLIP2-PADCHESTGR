# mymodels/blip2_lora.py

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model


def build_model_and_processor():
    """
    Loads BLIP-2 + LoRA and its processor.
    Returns: (model, processor)
    """
    # Processor
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Base model (8-bit, sharded)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "ybelkada/blip2-opt-2.7b-fp16-sharded",
        device_map="auto",
        load_in_8bit=True
    )

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.01,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, processor
