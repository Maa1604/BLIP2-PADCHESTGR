import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import Blip2ForConditionalGeneration, Blip2Processor

DEFAULT_PROMPTS = {
    "es": "Eres un asistente de radiología. Genera las conclusiones de la radiografía de tórax en español:\n",
    "en": "You are a radiology assistant. Generate chest X-ray findings in English:\n",
}

@dataclass
class Blip2PadChestConfig:
    checkpoint: str = "fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray"
    freeze_vision: bool = True        # often helpful to freeze the vision tower
    gradient_checkpointing: bool = True  # reduce memory during full FT
    dtype: str = "fp16"               # "fp16" | "bf16" | "fp32"

class Blip2PadChest(nn.Module):
    """
    Plain BLIP-2 wrapper (no LoRA):
      - Loads BLIP-2 with device_map='auto'
      - Optionally freezes the vision encoder
      - Exposes a Blip2Processor for training and generation
    """
    def __init__(self, cfg: Blip2PadChestConfig):
        super().__init__()
        self.cfg = cfg

        # Choose dtype
        if cfg.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif cfg.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Processor + Model (trainable weights; no 8-bit)
        self.processor: Blip2Processor = Blip2Processor.from_pretrained(cfg.checkpoint)
        self.model: Blip2ForConditionalGeneration = Blip2ForConditionalGeneration.from_pretrained(
            cfg.checkpoint,
            device_map="auto",
            torch_dtype=torch_dtype,
        )

        # Optional: freeze vision encoder
        if cfg.freeze_vision and hasattr(self.model, "vision_model"):
            for p in self.model.vision_model.parameters():
                p.requires_grad = False

        # Optional: gradient checkpointing to save memory
        if cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # Log trainable parameters
        try:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"[INFO] Trainable params: {trainable}/{total} ({100.0*trainable/total:.2f}%)")
            if cfg.freeze_vision and hasattr(self.model, "vision_model"):
                print("[INFO] Vision encoder is frozen.")
        except Exception:
            pass

    @property
    def device(self):
        # Returns first param's device for convenience
        return next(self.model.parameters()).device

    def forward(self, **batch):
        # batch contains: pixel_values, input_ids, attention_mask, labels
        return self.model(**batch)
