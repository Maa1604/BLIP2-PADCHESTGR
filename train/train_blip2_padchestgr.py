import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Any, List
import sys, os

# allow "python -m train.train_blip2_padchestgr" from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir)))

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from mydatasets.padchestgr_dataset import PadChestDataset
from mymodels.blip2_padchestgr import Blip2PadChest, Blip2PadChestConfig, DEFAULT_PROMPTS
from paths import DICT_CSV_PADCHESTGR_PATH  # reuse same dict (train/validation paths)

# ---- External scorers you already have ----
from myscorers.bleu.bleu import Bleu
from myscorers.rouge.rouge import Rouge
from myscorers.bertscore.bertscore import BertScorer
# from myscorers.chexbert.chexbert import myF1ChexBert
# from myscorers.myradgraph.myradgraph import myRadGraph


class Blip2Collator:
    """
    Builds prompt+target for BLIP-2 (OPT decoder), masking the prompt in labels.
    """
    def __init__(self, processor, lang: str):
        self.processor = processor
        self.lang = lang
        self.prompt = DEFAULT_PROMPTS[lang]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [f["image"] for f in features]
        targets = [f["text"] for f in features]
        prompts = [self.prompt for _ in targets]

        # Tokenize prompt+target, and mask the prompt part in labels
        text_inputs = self.processor.tokenizer(
            [p + t for p, t in zip(prompts, targets)],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        labels = text_inputs.input_ids.clone()

        for i in range(labels.size(0)):
            prompt_len = (prompt_inputs.input_ids[i] != self.processor.tokenizer.pad_token_id).sum()
            labels[i, :prompt_len] = -100  # ignore prompt in loss

        vision_inputs = self.processor(images=images, return_tensors="pt", padding=True)
        batch = {
            "pixel_values": vision_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
        }
        return batch


# ----------------- Metrics -----------------
bleu_scorer = Bleu(n=4)
rougel_scorer = Rouge(rouges=['rougeL'])
bert_scorer = BertScorer()
# f1cxb_scorer = myF1ChexBert()
# radgraph_scorer = myRadGraph(reward_level='partial')

@dataclass
class MetricBuffers:
    refs: List[str]
    hyps: List[str]

metric_buffers = MetricBuffers(refs=[], hyps=[])

def compute_metrics(_: EvalPrediction) -> Dict[str, float]:
    # handled in evaluate()
    return {}


from transformers.trainer import Trainer
class GenEvalTrainer(Trainer):
    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        model = self.model
        processor = self.args._processor
        lang = self.args._lang

        self.model.eval()
        metric_buffers.refs.clear(); metric_buffers.hyps.clear()

        ds = eval_dataset
        bs = self.args.per_device_eval_batch_size
        prompt = DEFAULT_PROMPTS[lang]

        # If model is wrapped, take its .device; else TrainingArguments' device
        device = getattr(model, "device", self.args.device)

        for i in range(0, len(ds), bs):
            subset = [ds[j] for j in range(i, min(i+bs, len(ds)))]
            images = [ex["image"] for ex in subset]
            refs = [ex["text"] for ex in subset]

            inputs = processor(images=images, text=[prompt]*len(images), return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_ids = model.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )
            hyps = processor.batch_decode(gen_ids, skip_special_tokens=True)
            hyps = [h.replace(prompt, "").strip() if h.startswith(prompt) else h.strip() for h in hyps]

            metric_buffers.refs.extend(refs)
            metric_buffers.hyps.extend(hyps)

        l_refs = [[r] for r in metric_buffers.refs]
        l_hyps = metric_buffers.hyps
        scores = {}
        try:
            scores.update({"bleu": bleu_scorer(l_refs, l_hyps)[0]})
        except Exception:
            pass
        try:
            scores.update({"rougeL": rougel_scorer(refs=l_refs, hyps=l_hyps)[0]})
        except Exception:
            pass
        try:
            bs_p, bs_r, bs_f = bert_scorer(l_hyps, l_refs)
            scores.update({"bertscore_f1": float(np.mean(bs_f))})
        except Exception:
            pass

        self.log({f"{metric_key_prefix}_{k}": v for k, v in scores.items()})
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray")
    parser.add_argument("--lang", type=str, default="es", choices=["es", "en"])
    parser.add_argument("--output_dir", type=str, default="EXPERIMENTS/BLIP2_PAD_BASE")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_freeze_vision", action="store_true")
    parser.add_argument("--no_grad_ckpt", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Datasets
    train_ds = PadChestDataset(DICT_CSV_PADCHESTGR_PATH["train"], lang=args.lang)
    val_ds = PadChestDataset(DICT_CSV_PADCHESTGR_PATH["validation"], lang=args.lang)

    # Model (no LoRA)
    dtype = "fp32"
    if args.bf16:
        dtype = "bf16"
    elif args.fp16:
        dtype = "fp16"

    cfg = Blip2PadChestConfig(
        checkpoint=args.checkpoint,
        freeze_vision=not args.no_freeze_vision,
        gradient_checkpointing=not args.no_grad_ckpt,
        dtype=dtype,
    )
    model_wrap = Blip2PadChest(cfg)

    data_collator = Blip2Collator(model_wrap.processor, args.lang)

    common_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        logging_steps=50,
        save_total_limit=3,
        report_to=["none"],
        remove_unused_columns=False,
        save_safetensors=False,
    )
    opt_kwargs = dict(
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_ratio=0.05,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        fp16=args.fp16,
        bf16=args.bf16,
    )

    training_args = TrainingArguments(**common_kwargs, **opt_kwargs)

    # pass a couple of helpers to trainer
    setattr(training_args, "_processor", model_wrap.processor)
    setattr(training_args, "_lang", args.lang)

    trainer = GenEvalTrainer(
        model=model_wrap,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_scores = trainer.evaluate(eval_dataset=val_ds)
    print("Final validation:", eval_scores)


if __name__ == "__main__":
    main()
