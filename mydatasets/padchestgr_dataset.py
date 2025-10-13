import os
import ast
from typing import Optional, Any, Dict, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from paths import IMAGES_PADCHESTGR_PATH  # keep your existing constant


def _parse_and_round_boxes_0_100(boxes_field: str) -> List[int]:
    """
    Parse the 'boxes' JSON-ish string and return a flat list of ints in [0..100],
    where each box [x1,y1,x2,y2] is multiplied by 100 and rounded.

    Expected structure per row:
      boxes = [[idx, "text", [[x1,y1,x2,y2], [..], ...]], ...]
    We collect *all* boxes in order and flatten them.
    """
    if not isinstance(boxes_field, str) or not boxes_field.strip():
        return []

    try:
        data = ast.literal_eval(boxes_field)
    except Exception:
        return []

    out = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        # item = [idx, "caption", [[x1,y1,x2,y2], ...]]
        boxes = item[2]
        if not isinstance(boxes, (list, tuple)):
            continue
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                for v in b:
                    try:
                        out.append(int(round(float(v) * 100)))
                    except Exception:
                        out.append(0)
    return out


class PadChestDataset(Dataset):
    """
    - Normal (grounded=False): returns PIL image + report text.
    - Grounded (grounded=True): additionally returns 'regions' as flat list of ints in [0..100].
    Also provides a build_collate_fn(processor) to batch examples.
    """
    def __init__(self, csv_path: str, image_root: Optional[str] = None, grounded: bool = False):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root or IMAGES_PADCHESTGR_PATH
        self.text_col = "report_en"
        self.grounded = grounded

        # minimal cleaning
        self.df = self.df.dropna(subset=["ImageID", self.text_col]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row["ImageID"])
        image = Image.open(image_path).convert("RGB")
        text = str(row[self.text_col]).strip()

        ex = {
            "image": image,
            "text": text,
            "image_id": row["ImageID"],
            "study_id": row.get("StudyID", ""),
        }

        if self.grounded:
            # parse normalized boxes -> ints in [0..100]
            regions = _parse_and_round_boxes_0_100(str(row.get("boxes", "")))
            ex["regions"] = regions  # flat list [x1,y1,x2,y2, x1,y1,x2,y2, ...] as ints

        return ex

    # ---------- collate fn builder ----------
    def build_collate_fn(self, processor: Any):
        """
        Returns a callable suitable for HuggingFace Trainer's data_collator.
        Uses the provided processor to batch images + texts and create labels.
        If grounded=True, also creates 'region_input_ids' by tokenizing a simple
        space-separated string of the rounded [0..100] box coordinates.
        """
        def _collate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            images = [ex["image"] for ex in examples]
            texts = [ex["text"] for ex in examples]

            batch = processor(
                images=images,
                text=texts,
                padding=True,
                return_tensors="pt"
            )

            # Create labels from input_ids, mask padding with -100
            input_ids = batch["input_ids"]
            labels = input_ids.clone()

            pad_id = processor.tokenizer.pad_token_id
            if pad_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
                pad_id = processor.tokenizer.pad_token_id

            labels[labels == pad_id] = -100
            batch["labels"] = labels

            # If grounded, build region_input_ids
            if self.grounded:
                # Build a simple string per sample like "52 55 80 83 51 54 79 84"
                region_texts = []
                for ex in examples:
                    coords = ex.get("regions", [])
                    if not coords:
                        region_texts.append("")  # empty → will tokenize to nothing
                    else:
                        region_texts.append(" ".join(str(v) for v in coords))

                tok = processor.tokenizer(
                    region_texts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                # LongTensor [B, W_len_tokens]
                batch["region_input_ids"] = tok["input_ids"]

            return batch

        return _collate