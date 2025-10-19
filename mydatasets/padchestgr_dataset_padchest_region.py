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


import math
from typing import Sequence, List

def _fixed_gaussian_points_from_box_0_100(
    box_xyxy: Sequence[int],
    *,
    n_points: int = 64,      # fixed target
    edge_margin: int = 1,    # keep at least this many pixels away from each box edge
    reach: float = 0.98,     # how close the outer ring gets to the allowed radius (0..1)
    rings_after_center: int = 7,  # 1 center + 7 rings = 8 layers total
) -> List[int]:
    """
    Generate exactly `n_points` integer (x,y) points inside a box [x1,y1,x2,y2] (0..100 ints).
    Points are arranged center-out in elliptical rings. The outermost ring reaches ~`reach`
    of the available radius but never touches the box edges (due to `edge_margin`).

    Returns a flat list [x,y,x,y,...] of length 2*n_points.
    """
    if not (isinstance(box_xyxy, (list, tuple)) and len(box_xyxy) == 4):
        return []

    x1, y1, x2, y2 = map(int, box_xyxy)
    # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    # order & clip to 0..100
    x1, x2 = max(0, min(x1, x2)), min(100, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(100, max(y1, y2))

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    # Degenerate (line/point). Just repeat center until reaching n_points.
    if w == 0 or h == 0:
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        return [cx, cy] * n_points

    # Box center in float
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # "Allowed" semi-axes inside the box after respecting margin on all sides
    # (half-width/height minus margin). Force >= 1 to avoid collapse.
    hx = max(1.0, (w / 2.0) - edge_margin)
    hy = max(1.0, (h / 2.0) - edge_margin)

    # Outer ring reaches `reach` of allowed semi-axes (keeps a small gap from the edges)
    rx_max = hx * reach
    ry_max = hy * reach

    # Distribute counts per ring (biased to put more points on outer rings),
    # but keep 1 point at exact center so there's always a center sample.
    # For n_points=64 and 7 rings after center → [1, 7, 8, 8, 10, 10, 10, 10] = 64.
    counts = [1] + [0] * rings_after_center
    remaining = n_points - 1
    # target weights for rings (1..rings_after_center): favor outer rings slightly
    weights = [i for i in range(1, rings_after_center + 1)]
    total_w = sum(weights)
    # initial proportional allocation
    alloc = [max(0, int(round(remaining * w / total_w))) for w in weights]
    # fix rounding drift
    diff = remaining - sum(alloc)
    # push or pull from outer rings first
    idxs = list(range(rings_after_center, 0, -1))  # outer→inner
    j = 0
    while diff != 0 and j < len(idxs) * 3:
        i = idxs[j % len(idxs)]
        k = i  # ring index in 1..rings_after_center
        if diff > 0:
            alloc[k-1] += 1
            diff -= 1
        else:
            if alloc[k-1] > 0:
                alloc[k-1] -= 1
                diff += 1
        j += 1
    # fill counts
    for i in range(1, rings_after_center + 1):
        counts[i] = alloc[i-1]

    # Radii for each ring (1..rings_after_center), center has radius 0
    # Use a mild convex ramp so rings spread fairly well toward the edges.
    R = float(rings_after_center)
    radii = [0.0]  # center
    for i in range(1, rings_after_center + 1):
        f = i / R
        # alpha slightly > 1 gives a touch more expansion near outer rings
        alpha = 1.2
        radii.append((f ** alpha))

    # Build points
    pts = []
    # center
    if counts[0] > 0:
        x_center = int(round(cx))
        y_center = int(round(cy))
        # clip inside margins
        x_center = min(max(x_center, x1 + edge_margin), x2 - edge_margin)
        y_center = min(max(y_center, y1 + edge_margin), y2 - edge_margin)
        pts.append((x_center, y_center))

    # rings
    for ring_idx in range(1, rings_after_center + 1):
        k = counts[ring_idx]
        if k <= 0:
            continue
        rfx = radii[ring_idx] * rx_max
        rfy = radii[ring_idx] * ry_max

        # Evenly spaced angles, small per-ring phase offset to avoid alignment
        # with image axes and reduce integer rounding collisions.
        phase = (ring_idx * 0.37) % 1.0
        for t in range(k):
            theta = (t / k + phase) * 2.0 * math.pi
            x = cx + rfx * math.cos(theta)
            y = cy + rfy * math.sin(theta)
            xi = int(round(x))
            yi = int(round(y))
            # Respect margins strictly
            xi = min(max(xi, x1 + edge_margin), x2 - edge_margin)
            yi = min(max(yi, y1 + edge_margin), y2 - edge_margin)
            pts.append((xi, yi))

    # Remove duplicates caused by rounding/clipping
    uniq = []
    seen = set()
    for xy in pts:
        if xy not in seen:
            uniq.append(xy)
            seen.add(xy)

    # If we lost some to dedup, top up by perturbing angles on the outermost ring
    # until we hit n_points or we run out of attempts.
    attempts = 0
    while len(uniq) < n_points and attempts < 200:
        # sample around the outer ring
        k = max(8, counts[-1])  # ensure enough angular resolution
        phase = ((attempts * 0.173) % 1.0)
        rfx = radii[-1] * rx_max
        rfy = radii[-1] * ry_max
        for t in range(k):
            if len(uniq) >= n_points:
                break
            theta = (t / k + phase) * 2.0 * math.pi
            x = cx + rfx * math.cos(theta)
            y = cy + rfy * math.sin(theta)
            xi = int(round(x))
            yi = int(round(y))
            xi = min(max(xi, x1 + edge_margin), x2 - edge_margin)
            yi = min(max(yi, y1 + edge_margin), y2 - edge_margin)
            if (xi, yi) not in seen:
                uniq.append((xi, yi))
                seen.add((xi, yi))
        attempts += 1

    # If we somehow still have fewer than n_points (very tiny boxes), fill by scanning the box
    if len(uniq) < n_points:
        for yy in range(y1 + edge_margin, y2 - edge_margin + 1):
            for xx in range(x1 + edge_margin, x2 - edge_margin + 1):
                if (xx, yy) not in seen:
                    uniq.append((xx, yy))
                    seen.add((xx, yy))
                    if len(uniq) >= n_points:
                        break
            if len(uniq) >= n_points:
                break

    # Truncate (or assert) to exactly n_points
    uniq = uniq[:n_points]

    # Flatten
    out: List[int] = []
    for (xx, yy) in uniq:
        out.extend((xx, yy))
    
    # print(f"out: {out}")
    # exit()
    return out

import ast
from typing import List

def _boxes_to_gaussian_points_0_100(
    boxes_field: str,
    *,
    points_per_box: int = 64,
    edge_margin: int = 1,
    reach: float = 0.98,
) -> List[int]:
    """
    Parse 'boxes' (normalized 0..1), and for each [x1,y1,x2,y2] produce
    exactly `points_per_box` integer (x,y) points (flat list) using
    _fixed_gaussian_points_from_box_0_100. No global cap; fixed per-box budget.
    """
    if not isinstance(boxes_field, str) or not boxes_field.strip():
        return []

    try:
        data = ast.literal_eval(boxes_field)
    except Exception:
        return []

    out_xy: List[int] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        boxes = item[2]
        if not isinstance(boxes, (list, tuple)):
            continue
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                # normalized → 0..100 ints
                try:
                    bx = [int(round(float(v) * 100)) for v in b]
                except Exception:
                    bx = [0, 0, 0, 0]
                pts = _fixed_gaussian_points_from_box_0_100(
                    bx,
                    n_points=points_per_box,
                    edge_margin=edge_margin,
                    reach=reach,
                )
                out_xy.extend(pts)

    return out_xy


def _format_region_points(points, *, trailing_comma: bool = True) -> str:
    """
    Format a flat list like [x1,y1,x2,y2,...] into:
    "[ x1 y1 ] [ x2 y2 ] ..." (optionally ending with ',')
    """
    # Pair up: (x, y); drop last if odd-length
    it = iter(points)
    pairs = []
    try:
        while True:
            x = next(it)
            y = next(it)
            pairs.append((x, y))
    except StopIteration:
        pass  # odd-length → last value is ignored

    if not pairs:
        return ""  # or "[]" if you prefer a visible empty

    s = " ".join(f"[ {int(x)} {int(y)} ]" for x, y in pairs)
    return s + " ," if trailing_comma else s


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
            regions = _boxes_to_gaussian_points_0_100(
                str(row.get("boxes", "")),
            )
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
                    if coords:
                        region_texts.append(_format_region_points(coords, trailing_comma=False))
                    else:
                        region_texts.append("")
                    # print(f"region_texts: {region_texts}")

                tok = processor.tokenizer(
                    region_texts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                batch["region_input_ids"] = tok["input_ids"]  # LongTensor [B, W_len_tokens]

            return batch

        return _collate
