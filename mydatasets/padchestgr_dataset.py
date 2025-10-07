import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional

from paths import IMAGES_PADCHESTGR_PATH  # keep your existing constant

class PadChestDataset(Dataset):
    """
    ROI-free PadChest dataset for BLIP-2 baseline.
    Returns dicts with PIL image + final text.
    """
    def __init__(self, csv_path: str, lang: str = "es", image_root: Optional[str] = None):
        assert lang in {"es", "en"}
        self.lang = lang
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root or IMAGES_PADCHESTGR_PATH
        self.text_col = "report_es" if lang == "es" else "report_en"

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
