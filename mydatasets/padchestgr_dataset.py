import os
from typing import Optional, Any, Dict, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from paths import IMAGES_PADCHESTGR_PATH  # keep your existing constant


class PadChestDataset(Dataset):
    """
    ROI-free PadChest dataset for BLIP-2 baseline.
    Returns dicts with PIL image + final TEXT.
    Also provides a build_collate_fn(processor) to batch examples.
    """
    def __init__(self, csv_path: str, image_root: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root or IMAGES_PADCHESTGR_PATH
        self.text_col = "report_en"

        # minimal cleaning
        self.df = self.df.dropna(subset=["ImageID", self.text_col]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row["ImageID"])
        image = Image.open(image_path).convert("RGB")
        text = str(row[self.text_col]).strip()
        return {
            "image": image,
            "text": text,
            "image_id": row["ImageID"],
            "study_id": row.get("StudyID", ""),
        }

    # ---------- NEW: collate fn builder ----------
    def build_collate_fn(self, processor: Any):
        """
        Returns a callable suitable for HuggingFace Trainer's data_collator.
        It uses the provided processor to batch images + texts and create labels.
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
                # Ensure pad token exists (edge case)
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
                pad_id = processor.tokenizer.pad_token_id

            labels[labels == pad_id] = -100
            batch["labels"] = labels

            return batch

        return _collate
