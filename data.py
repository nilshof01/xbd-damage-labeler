"""xBD data loading and stratified ordering."""
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

from PIL import Image

HF_DATASET_REPO = "nilsho01/xbd-train"


def get_xbd_dir() -> Path | None:
    """Return local xBD train directory, or None if running from HF Hub."""
    local = os.environ.get("XBD_DIR", "")
    if local and Path(local).exists():
        return Path(local)
    return None  # HF Spaces: load images lazily via hf_hub_download

EVENT_COUNTRY = {
    "guatemala-volcano":   "Guatemala",
    "hurricane-florence":  "USA",
    "hurricane-harvey":    "USA",
    "hurricane-matthew":   "Haiti",
    "hurricane-michael":   "USA",
    "mexico-earthquake":   "Mexico",
    "midwest-flooding":    "USA",
    "palu-tsunami":        "Indonesia",
    "santa-rosa-wildfire": "USA",
    "socal-fire":          "USA",
}


@dataclass
class Tile:
    id: str           # e.g. "hurricane-matthew_00000123"
    event: str
    country: str
    pre_date: str
    post_date: str
    pre_image_path: Path
    post_image_path: Path
    pre_label_path: Path
    post_label_path: Path


def scan_tiles(xbd_dir: str | Path | None) -> list[Tile]:
    if xbd_dir is not None:
        return _scan_local(Path(xbd_dir))
    return _scan_hf()


def _scan_local(xbd_dir: Path) -> list[Tile]:
    labels_dir = xbd_dir / "labels"
    images_dir = xbd_dir / "images"

    pre_labels  = {p.stem.replace("_pre_disaster",  ""): p
                   for p in labels_dir.glob("*_pre_disaster.json")}
    post_labels = {p.stem.replace("_post_disaster", ""): p
                   for p in labels_dir.glob("*_post_disaster.json")}

    tiles = []
    for tile_id, pre_lbl in sorted(pre_labels.items()):
        if tile_id not in post_labels:
            continue
        event   = tile_id.split("_")[0]
        country = EVENT_COUNTRY.get(event, "Unknown")
        pre_img  = images_dir / f"{tile_id}_pre_disaster.png"
        post_img = images_dir / f"{tile_id}_post_disaster.png"
        if not pre_img.exists() or not post_img.exists():
            continue
        tiles.append(Tile(
            id=tile_id, event=event, country=country,
            pre_date=_read_date(pre_lbl),
            post_date=_read_date(post_labels[tile_id]),
            pre_image_path=pre_img,
            post_image_path=post_img,
            pre_label_path=pre_lbl,
            post_label_path=post_labels[tile_id],
        ))
    return tiles


def _scan_hf() -> list[Tile]:
    """Build tile list from HF Hub file listing (no local data needed)."""
    from huggingface_hub import list_repo_files, hf_hub_download
    all_files = list(list_repo_files(HF_DATASET_REPO, repo_type="dataset"))

    pre_imgs  = {f.split("/")[-1].replace("_pre_disaster.png",  "")
                 for f in all_files if f.endswith("_pre_disaster.png")}
    post_imgs = {f.split("/")[-1].replace("_post_disaster.png", "")
                 for f in all_files if f.endswith("_post_disaster.png")}

    tiles = []
    for tile_id in sorted(pre_imgs & post_imgs):
        event   = tile_id.split("_")[0]
        country = EVENT_COUNTRY.get(event, "Unknown")
        # Download label JSONs to read dates (small files, fast)
        try:
            pre_lbl  = Path(hf_hub_download(HF_DATASET_REPO, repo_type="dataset",
                                             filename=f"labels/{tile_id}_pre_disaster.json"))
            post_lbl = Path(hf_hub_download(HF_DATASET_REPO, repo_type="dataset",
                                             filename=f"labels/{tile_id}_post_disaster.json"))
        except Exception:
            continue
        # Use dummy paths — actual image loading goes through hf_hub_download
        dummy = Path("/nonexistent")
        tiles.append(Tile(
            id=tile_id, event=event, country=country,
            pre_date=_read_date(pre_lbl),
            post_date=_read_date(post_lbl),
            pre_image_path=dummy,
            post_image_path=dummy,
            pre_label_path=pre_lbl,
            post_label_path=post_lbl,
        ))
    return tiles


def stratify_tiles(tiles: list[Tile], seed: int = 42) -> list[Tile]:
    """Round-robin across events so early tiles are geographically diverse."""
    rng = random.Random(seed)
    by_event: dict[str, list[Tile]] = defaultdict(list)
    for t in tiles:
        by_event[t.event].append(t)
    for lst in by_event.values():
        rng.shuffle(lst)

    result, active = [], list(by_event.keys())
    rng.shuffle(active)
    iters = {e: iter(lst) for e, lst in by_event.items()}
    while active:
        for e in list(active):
            try:
                result.append(next(iters[e]))
            except StopIteration:
                active.remove(e)
    return result


def load_image_pair(tile: Tile) -> tuple[Image.Image, Image.Image]:
    if tile.pre_image_path.exists():
        pre  = Image.open(tile.pre_image_path).convert("RGB")
        post = Image.open(tile.post_image_path).convert("RGB")
        return pre, post
    # HF Hub lazy fetch — downloads once, then cached in ~/.cache/huggingface
    from huggingface_hub import hf_hub_download
    pre_path = hf_hub_download(
        repo_id=HF_DATASET_REPO, repo_type="dataset",
        filename=f"images/{tile.id}_pre_disaster.png")
    post_path = hf_hub_download(
        repo_id=HF_DATASET_REPO, repo_type="dataset",
        filename=f"images/{tile.id}_post_disaster.png")
    return Image.open(pre_path).convert("RGB"), Image.open(post_path).convert("RGB")


def _read_date(label_path: Path) -> str:
    try:
        with open(label_path) as f:
            meta = json.load(f).get("metadata", {})
        return meta.get("capture_date", meta.get("date", ""))[:10]
    except Exception:
        return ""
