import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import LoraConfig, get_peft_model, TaskType

DEFAULT_PROMPTS = {
    "es": "Eres un asistente de radiología. Genera las conclusiones de la radiografía de tórax en español:\n",
    "en": "You are a radiology assistant. Generate chest X-ray findings in English:\n",
}

@dataclass
class Blip2PadchestLoRA:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules_text: Optional[List[str]] = None
    target_modules_qformer: Optional[List[str]] = None

@dataclass
class Blip2PadChestConfig:
    checkpoint: str = "fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray"
    use_lora: bool = True
    lora: Blip2PadchestLoRA = Blip2PadchestLoRA()
    freeze_vision: bool = True  # often helpful

class Blip2PadChest(nn.Module):
    """
    ROI-free BLIP-2 wrapper:
      - Loads BLIP-2 in 8-bit with device_map='auto'
      - Optionally injects LoRA if the checkpoint doesn't already have adapters
      - Keeps a Blip2Processor for both training and generation
    """
    def __init__(self, cfg: Blip2PadChestConfig):
        super().__init__()
        self.cfg = cfg

        # Processor + Model in the "plain" BLIP-2 style (as requested)
        self.processor: Blip2Processor = Blip2Processor.from_pretrained(cfg.checkpoint)
        self.model: Blip2ForConditionalGeneration = Blip2ForConditionalGeneration.from_pretrained(
            cfg.checkpoint,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

        if cfg.freeze_vision and hasattr(self.model, "vision_model"):
            for p in self.model.vision_model.parameters():
                p.requires_grad = False

        # If checkpoint already includes PEFT adapters, just make those trainable.
        has_existing_adapters = hasattr(self.model, "peft_config") and isinstance(self.model.peft_config, dict) and len(self.model.peft_config) > 0
        if has_existing_adapters:
            print("[INFO] Found existing PEFT adapters in checkpoint. Training those; not injecting new LoRA.")
            for p in self.model.parameters():
                p.requires_grad = False
            for n, p in self.model.named_parameters():
                if "lora_" in n or "adapter" in n:
                    p.requires_grad = True
            try:
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"[INFO] Trainable params in existing adapters: {trainable}/{total}")
            except Exception:
                pass
        elif cfg.use_lora:
            # Inject LoRA on Q-Former + OPT layers
            qf_targets = cfg.lora.target_modules_qformer or [
                "qformer.decoder.layers.*.self_attn.q_proj",
                "qformer.decoder.layers.*.self_attn.k_proj",
                "qformer.decoder.layers.*.self_attn.v_proj",
                "qformer.decoder.layers.*.self_attn.out_proj",
            ]
            txt_targets = cfg.lora.target_modules_text or [
                "language_model.model.decoder.layers.*.self_attn.q_proj",
                "language_model.model.decoder.layers.*.self_attn.k_proj",
                "language_model.model.decoder.layers.*.self_attn.v_proj",
                "language_model.model.decoder.layers.*.self_attn.out_proj",
                "language_model.model.decoder.layers.*.fc1",
                "language_model.model.decoder.layers.*.fc2",
            ]
            peft_cfg = LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=qf_targets + txt_targets,
            )
            self.model = get_peft_model(self.model, peft_cfg)
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass

    @property
    def device(self):
        # PEFT/model may be sharded; this returns first param's device for convenience
        return next(self.model.parameters()).device

    def forward(self, **batch):
        # batch contains: pixel_values, input_ids, attention_mask, labels
        return self.model(**batch)
