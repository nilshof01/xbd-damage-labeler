"""UNet building instance segmentation on xBD 1024x1024 tiles.

Model: nilsho01/unet-buildings-autorun-v2 (SMP UNet, EfficientNet-B4 encoder).
Sliding-window inference: 256x256 patches, 4x4 grid over 1024x1024.
Instances extracted via connected components; small blobs filtered out.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

MODEL_REPO   = "nilsho01/unet-buildings-autorun-v2"
PATCH_SIZE   = 256
STRIDE       = 256      # no overlap — fast and sufficient at 1024x1024
MIN_AREA_PX  = 30       # drop instances smaller than this (noise)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: str = "cpu"):
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file

    weights_path = hf_hub_download(MODEL_REPO, filename="model.safetensors")
    # Try dist4 head first (4 output classes), fall back to binary (1 class)
    for n_out in (4, 2, 1):
        try:
            net = smp.Unet(
                encoder_name="efficientnet-b4",
                encoder_weights=None,
                in_channels=3,
                classes=n_out,
            )
            state = load_file(weights_path)
            net.load_state_dict(state, strict=True)
            net.eval().to(device)
            return net, n_out
        except Exception:
            continue

    raise RuntimeError(
        f"Could not load {MODEL_REPO}. "
        "Check that segmentation_models_pytorch is installed and the "
        "model architecture matches."
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def _preprocess(patch: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> 1x3xHxW float32 normalised to ImageNet stats."""
    x = patch.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def predict_mask(img_pil: Image.Image,
                 model: torch.nn.Module,
                 n_out: int,
                 device: str = "cpu",
                 threshold: float = 0.5) -> np.ndarray:
    """Return a boolean (H, W) building mask for a 1024x1024 image."""
    img = np.array(img_pil.convert("RGB"))
    H, W = img.shape[:2]
    prob = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    with torch.inference_mode():
        for y in range(0, H, STRIDE):
            for x in range(0, W, STRIDE):
                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                ph, pw = patch.shape[:2]
                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    # pad to PATCH_SIZE
                    padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                tensor = _preprocess(patch).to(device)
                out = model(tensor)            # (1, n_out, H, W)
                # channel 0 is always the mask logit
                logit = out[0, 0, :ph, :pw].cpu().numpy()
                p = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
                prob[y:y+ph, x:x+pw] += p
                count[y:y+ph, x:x+pw] += 1.0

    prob /= np.maximum(count, 1.0)
    return prob > threshold


# ── Instance extraction ───────────────────────────────────────────────────────

def extract_instances(binary_mask: np.ndarray,
                      min_area: int = MIN_AREA_PX
                      ) -> tuple[np.ndarray, list[dict]]:
    """Connected components -> labeled array + list of instance dicts.

    Returns:
        labeled   (H, W) int32 — 0 = background, 1..N = building instances
        instances list of dicts with keys: idx, centroid_x, centroid_y, area
    """
    from scipy import ndimage

    labeled, n = ndimage.label(binary_mask)
    instances = []
    new_labeled = np.zeros_like(labeled)
    new_idx = 1

    for i in range(1, n + 1):
        mask_i = labeled == i
        area = int(mask_i.sum())
        if area < min_area:
            continue
        cy, cx = ndimage.center_of_mass(mask_i)
        new_labeled[mask_i] = new_idx
        instances.append({
            "idx":        new_idx,
            "centroid_x": int(round(cx)),
            "centroid_y": int(round(cy)),
            "area":       area,
        })
        new_idx += 1

    return new_labeled.astype(np.int32), instances


# ── Visualisation ─────────────────────────────────────────────────────────────

DAMAGE_COLORS = {
    "intact":            (80,  200,  80),
    "severely_damaged": (255, 165,   0),
    "destroyed":        (220,  50,  50),
}
BORDER_WIDTH = 2


def draw_instance_borders(img_pil: Image.Image,
                           labeled: np.ndarray,
                           states: dict[int, str]) -> Image.Image:
    """Draw coloured building borders on img_pil based on damage states.

    Args:
        img_pil  : original image
        labeled  : (H, W) int32 instance map
        states   : {instance_idx: damage_state_str}  (missing = 'intact')
    """
    from skimage.segmentation import find_boundaries

    img = np.array(img_pil.convert("RGB")).copy()

    unique_ids = np.unique(labeled)
    unique_ids = unique_ids[unique_ids > 0]

    for idx in unique_ids:
        state = states.get(int(idx), "intact")
        color = DAMAGE_COLORS[state]
        mask_i = labeled == idx
        border = find_boundaries(mask_i, mode="outer")
        for _ in range(BORDER_WIDTH):
            img[border, 0] = color[0]
            img[border, 1] = color[1]
            img[border, 2] = color[2]

    return Image.fromarray(img)
