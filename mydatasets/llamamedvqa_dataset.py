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
    
    @staticmethod
    def _split_csv_field(val: Optional[str]) -> List[str]:
        """
        Split a comma-separated cell into a list, trimming whitespace.
        Robust to NaN/None. Does NOT attempt to parse nested commas in paths
        because the CSV example uses quotes correctly.
        """
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        # ensure string, strip surrounding quotes, then split
        s = str(val).strip().strip('"').strip("'")
        # allow both comma and semicolon just in case
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip() != ""]
        return parts if parts else []

    @staticmethod
    def _choose_index_by_view(views: List[str]) -> int:
        """
        Return the first index whose view is AP or PA (case-insensitive).
        If none found or views is empty, return 0.
        """
        targets = {"ap", "pa"}
        for i, v in enumerate(views):
            if str(v).strip().lower() in targets:
                return i
        return 0

    def _resolve_image_path(self, rel_or_abs: str) -> str:
        """Return absolute path for an image path that may be relative to image_root."""
        return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(self.image_root, rel_or_abs)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]


        # Parse image paths (may be one or many)
        img_cells = self._split_csv_field(row["image_path"])

        # Parse view positions (aligned order if present)
        views = self._split_csv_field(row.get("view_position", None))

        # Choose index: first AP/PA if possible, else 0
        pick_idx = self._choose_index_by_view(views) if img_cells else 0
        if pick_idx >= len(img_cells):
            # views length may exceed or not match images; clamp
            pick_idx = 0

        chosen_rel = img_cells[pick_idx] if img_cells else str(row["image_path"])
        chosen_path = self._resolve_image_path(chosen_rel)
        
        # Load image
        image = Image.open(chosen_path).convert("RGB")

        # Texts
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        prompt = self.prompt_template.format(q=question)

        # Optional: keep which view we picked (helpful for debugging/analysis)
        # chosen_view = views[pick_idx] if views and pick_idx < len(views) else None

        # print(f"image_path: {chosen_rel}")
        # # print(f"chosen_view: {chosen_view}")
        # print(f"question: {question}")
        # print(f"answer: {answer}")
        # print(f"prompt: {prompt}")
        # print(f"-------------\n")
        # exit()

        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "image_path": chosen_rel,
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
