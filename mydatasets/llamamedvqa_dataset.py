import os
from typing import Optional, Any, Dict, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from paths import IMAGES_MIMIC_PATH  # keep your existing constant

VQA_PROMPT_TEMPLATE = "Question: {q} Answer:"

class LlamaMedVQADataset(Dataset):
    """
    VQA dataset:
      - columns: image_path, question, answer
      - __getitem__ returns PIL image + (prompt, answer) + ids
      - build_collate_fn(processor, ...) creates encoder inputs from prompt and decoder labels from answer
    """
    def __init__(self, csv_path: str, image_root: Optional[str] = None, prompt_template: str = VQA_PROMPT_TEMPLATE):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root or IMAGES_MIMIC_PATH
        self.prompt_template = prompt_template

        # minimal cleaning
        self.df = self.df.dropna(subset=["image_path", "question", "answer"]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # image path can be absolute or relative to image_root
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        image = Image.open(img_path).convert("RGB")
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        prompt = self.prompt_template.format(q=question)

        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "image_path": row["image_path"],
            "question": question,
        }

    # ---------- collate fn ----------
    def build_collate_fn(self, processor: Any):
        """
        Returns a callable that:
          - tokenizes prompts for the encoder (inputs)
          - tokenizes answers as decoder labels (T5-style), pads to -100
        """
        def _collate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            images  = [ex["image"] for ex in examples]
            prompts = [ex["prompt"] for ex in examples]
            answers = [ex["answer"] for ex in examples]

            # 1) Encoder inputs (image + prompt)
            enc = processor(
                images=images,
                text=prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # 2) Decoder targets (answers only)
            # Use the tokenizer directly so labels are just the expected answer tokens.
            tok = processor.tokenizer
            tgt = tok(
                answers,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            labels = tgt.input_ids
            pad_id = tok.pad_token_id
            if pad_id is None:
                tok.pad_token = tok.eos_token
                pad_id = tok.pad_token_id
            labels[labels == pad_id] = -100

            enc["labels"] = labels

            # Keep strings for metrics/logging
            enc["references"]   = answers          # gold answers
            enc["prompts"]      = prompts          # the question prompts
            enc["questions"]    = [ex["question"] for ex in examples]
            enc["image_paths"]  = [ex["image_path"] for ex in examples]

            return enc

        return _collate
