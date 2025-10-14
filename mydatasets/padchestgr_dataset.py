import os
import ast
from typing import Optional, Any, Dict, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

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


def _open_image_as_8bit_rgb(path: str, p: float = 99.0) -> Image.Image:
    """
    Open an image and normalize it to 8-bit RGB using a p-th percentile cap.
    """
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)

    if arr.size == 0 or not np.isfinite(arr).any():
        return img.convert("RGB")

    pth = np.nanpercentile(arr, p)
    if not np.isfinite(pth) or pth <= 0:
        pth = 1.0

    arr = np.clip(arr / (pth + 1e-8), 0.0, 1.0)
    arr8 = (arr * 255.0).astype(np.uint8)

    if arr8.ndim == 2:
        rgb = np.stack([arr8, arr8, arr8], axis=-1)
    elif arr8.ndim == 3:
        if arr8.shape[-1] == 3:
            rgb = arr8
        elif arr8.shape[-1] == 4:
            rgb = arr8[..., :3]
        else:
            g = arr8.mean(axis=-1).astype(np.uint8)
            rgb = np.stack([g, g, g], axis=-1)
    else:
        return img.convert("RGB")

    return Image.fromarray(rgb, mode="RGB")


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
        image = _open_image_as_8bit_rgb(image_path)
        text = str(row[self.text_col]).strip()

        ex = {
            "image": image,
            "text": text,
            "image_id": row["ImageID"],
            "study_id": row.get("StudyID", ""),
        }

        if self.grounded:
            regions = _parse_and_round_boxes_0_100(str(row.get("boxes", "")))
            ex["regions"] = regions  # flat list [x1,y1,x2,y2, ...] as ints

        # print(f"image: {ex['image']}")
        # print(f"text: {ex['text']}")
        # print(f"image_id: {ex['image_id']}")
        # print(f"study_id: {ex['study_id']}")
        # print(f"regions: {ex.get('regions', '')}")

        return ex

    # ---------- collate fn builder ----------
    def build_collate_fn(self, processor: Any):
        """
        Returns a callable suitable for HuggingFace Trainer's data_collator.
        """
        def _collate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            images = [ex["image"] for ex in examples]
            texts = [ex["text"] for ex in examples]
            image_ids = [ex["image_id"] for ex in examples]

            batch = processor(
                images=images,
                text=texts,
                padding=True,
                return_tensors="pt",
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

            # Keep texts and ids for evaluation/logging
            batch["references"] = texts  # list[str] for scorers
            batch["image_ids"] = image_ids  # list[str]

            # If grounded, build region_input_ids
            if self.grounded:
                region_texts = []
                for ex in examples:
                    coords = ex.get("regions", [])
                    if not coords:
                        region_texts.append("")
                    else:
                        region_texts.append(" ".join(str(v) for v in coords))

                tok = processor.tokenizer(
                    region_texts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                batch["region_input_ids"] = tok["input_ids"]  # LongTensor [B, W_len_tokens]

            return batch

        return _collate
