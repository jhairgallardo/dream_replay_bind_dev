import math
import random
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

def collate_function(batch):
    """
    batch = list of tuples returned by __getitem__
    0) stack *views*  -> Tensor (B, V, 3, H, W)
    1) keep *aug_lists* as pure Python list-of-lists
    """
    views_b     = torch.stack([item[0][0] for item in batch], dim=0)
    aug_lists_b = [item[0][1] for item in batch]   # len = B
    labels_b    = torch.tensor([item[1] for item in batch]) # labels\
    taskid_b   = torch.tensor([item[2] for item in batch]) # taskid
    return (views_b, aug_lists_b), labels_b , taskid_b

def collate_function_notaskid(batch):
    """
    batch = list of tuples returned by __getitem__
    0) stack *views*  -> Tensor (B, V, 3, H, W)
    1) keep *aug_lists* as pure Python list-of-lists
    """
    views_b     = torch.stack([item[0][0] for item in batch], dim=0)
    aug_lists_b = [item[0][1] for item in batch]   # len = B
    labels_b    = torch.tensor([item[1] for item in batch]) # labels\
    return (views_b, aug_lists_b), labels_b
    

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _normalize_sincos(sin_th: float, cos_th: float) -> Tuple[float, float]:
    """Ensure (sin, cos) lies on the unit circle (robust to small numeric drift)."""
    mag = math.hypot(cos_th, sin_th)
    if mag == 0.0:
        return 0.0, 1.0  # default to angle 0
    return sin_th / mag, cos_th / mag


def _zoom_to_side_px(zoom: float, W: int, H: int) -> int:
    """
    zoom = area_crop / area_image  (area_image = W*H)
    side_px = round(sqrt(zoom * W * H)), with a minimum of 1.
    """
    z = _clamp(float(zoom), 0.0, 1.0)
    side = int(round(math.sqrt(z * W * H)))
    return max(1, min(W, min(H, side)))


def _max_R_allowed(sin_th: float, cos_th: float, side_px: int, W: int, H: int) -> float:
    """
    Given a direction (sinθ, cosθ) and crop side, return the largest R (pixels)
    such that the square crop centered at (W/2, H/2) + R*(cosθ, sinθ) with half-side s=side_px/2
    stays fully inside the image.

    Constraints:
      Cx = W/2 + R*cosθ in [s, W - s]  =>  |R * cosθ| <= (W/2 - s)
      Cy = H/2 + R*sinθ in [s, H - s]  =>  |R * sinθ| <= (H/2 - s)

    Thus:
      if cosθ != 0: R <= (W/2 - s) / |cosθ| else x-bound = +inf
      if sinθ != 0: R <= (H/2 - s) / |sinθ| else y-bound = +inf

    Also cap by half-diagonal (distance to corners), R_hd = hypot(W/2, H/2).
    """
    sin_th, cos_th = _normalize_sincos(sin_th, cos_th)
    s = side_px / 2.0
    half_w, half_h = W / 2.0, H / 2.0

    # If the crop already fills the shorter dimension, the center must be exactly at the image center.
    margin_x = max(0.0, half_w - s)
    margin_y = max(0.0, half_h - s)

    # Directional bounds
    eps = 1e-12
    if abs(cos_th) > eps:
        R_x = margin_x / abs(cos_th)
    else:
        R_x = float("inf")

    if abs(sin_th) > eps:
        R_y = margin_y / abs(sin_th)
    else:
        R_y = float("inf")

    # Global geometric cap: half-diagonal
    R_hd = math.hypot(half_w, half_h)

    R_lim = min(R_x, R_y, R_hd)
    return max(0.0, R_lim)


def _center_from_R_px(sin_th: float, cos_th: float, R_px: float, W: int, H: int) -> Tuple[float, float]:
    """Compute (cx, cy) in pixels from a radius R along direction (cosθ, sinθ) from the image center."""
    sin_th, cos_th = _normalize_sincos(sin_th, cos_th)
    half_w, half_h = W / 2.0, H / 2.0
    cx = half_w + R_px * cos_th
    cy = half_h + R_px * sin_th
    return cx, cy


def _center_px_to_box(cx: float, cy: float, side_px: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """
    Convert center (pixels) and integer side to a torchvision crop box (top, left, height, width).
    """
    w = h = int(side_px)
    left = int(round(cx - w / 2.0))
    top  = int(round(cy - h / 2.0))
    left = _clamp(left, 0, W - w)
    top  = _clamp(top,  0, H - h)
    return top, left, h, w


class Episode_Transformations:
    """
    Episode of V square crops from an image, each defined by polar params w.r.t. the image center.

    For each view t we sample:
      - zoom_t ∈ [0,1]: area ratio; side_px = round(sqrt(zoom_t * W * H))
      - [sinθ_t, cosθ_t]: direction from image center
      - R_t (pixels): sampled ∈ [0, R_limit], where R_limit respects borders for (side_px, θ)
        and is also capped by the half-diagonal.
      - r_t = R_t / R_hd, where R_hd = hypot(W/2, H/2) = 112*sqrt(2) for 224×224

      center: (cx, cy) = (W/2, H/2) + R_t * (cosθ_t, sinθ_t)

    The action for that view is:
        action_t = [sinθ_t, cosθ_t, r_t, zoom_t]

    Returns:
      views   : Tensor (V, 3, size, size)
      actions : List[torch.tensor(4,)] of length V  (one action per *current* view)
    """

    def __init__(
        self,
        num_views: int = 4,
        size: int = 224,                 # output crop size
        geom_size: int = 224,            # we compute geometry on a 224×224 canvas (your spec)
        mean: List[float] = [0.5, 0.5, 0.5],
        std:  List[float] = [0.5, 0.5, 0.5],
        zoom_range: Tuple[float, float] = (0.05, 0.85),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        assert num_views > 0
        self.num_views = int(num_views)
        self.size = int(size)
        self.geom_size = int(geom_size)
        self.mean, self.std = mean, std
        self.zoom_range = (float(zoom_range[0]), float(zoom_range[1]))
        self.interp = interpolation

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    # ---- simple samplers (customize for temporal coherence if you want) ---- #
    def _sample_zoom(self) -> float:
        z0, z1 = self.zoom_range
        z0 = _clamp(z0, 0.0, 1.0)
        z1 = _clamp(z1, 0.0, 1.0)
        if z0 > z1:
            z0, z1 = z1, z0
        return random.uniform(z0, z1)

    def _sample_theta(self) -> Tuple[float, float]:
        th = random.uniform(-math.pi, math.pi)
        return math.sin(th), math.cos(th)

    # ----------------------------------------------------------------------- #

    def _ensure_geom_image(self, img: Image.Image) -> Image.Image:
        """Ensure the geometry canvas is 224×224 (or geom_size×geom_size)."""
        if img.size != (self.geom_size, self.geom_size):
            return img.resize((self.geom_size, self.geom_size), resample=Image.BILINEAR)
        return img

    def _random_crop_image(self, base_img: Image.Image) -> Image.Image:
        
        W, H = base_img.size  # should be 224,224

        # Half-diagonal used for r normalization (constant for fixed W,H)
        R_hd = math.hypot(W / 2.0, H / 2.0)

        # 1) sample zoom and angle
        zoom = self._sample_zoom()
        sin_th, cos_th = self._sample_theta()

        # 2) compute side in pixels from zoom
        side_px = _zoom_to_side_px(zoom, W, H)

        # 3) compute direction-aware, size-aware max R, then sample R
        R_limit = _max_R_allowed(sin_th, cos_th, side_px, W, H)  # pixels
        if R_limit <= 0.0:
            R = 0.0
        else:
            R = random.uniform(0.0, R_limit)

        # 4) normalized radius r for the action payload
        r_norm = 0.0 if R_hd == 0 else (R / R_hd)
        # Clamp just in case of tiny FP overshoot
        r_norm = _clamp(r_norm, 0.0, 1.0)

        # 5) center and crop
        cx, cy = _center_from_R_px(sin_th, cos_th, R, W, H)
        top, left, h, w = _center_px_to_box(cx, cy, side_px, W, H)

        crop = F.resized_crop(base_img, top, left, h, w, [self.size, self.size], interpolation=self.interp)
        act = torch.tensor([sin_th, cos_th, r_norm, zoom], dtype=torch.float32)

        return crop, act



    def __call__(self, pil_img: Image.Image):
        assert isinstance(pil_img, Image.Image), "Input must be a PIL.Image"
        base_img = self._ensure_geom_image(pil_img)  # enforce 224×224 geometry

        views = torch.zeros(self.num_views, 3, base_img.size[0], base_img.size[1])
        actions = []

        for i in range(self.num_views):
            view_actions = []

            view, action_cropbb = self._random_crop_image(base_img)
            view_actions.append(("crop", action_cropbb))

            # Todo:
            # Horizontal flip: 1 means active, 0 means inactive
            # Color jitter: for inactive use the set of values the mean that nothing is done.
            # threaugment
            #    if choice 0 -> grayscale p=1, blur p=0, solar p =0
            #    if choice 1 -> grayscale p=0, blur p=1, solar p =0
            #    if choice 2 -> grayscale p=0, blur p=0, solar p =1
            # With that way, I can get the action parameters for the unselected cases
            # For example, for grayscale, 1 means active, 0 means inactive
            # For blur, it is based on the sigma value. A very low sigma basically means no blur
            # For solar, it is based on the ratio of pixels that are solarized to the total number of pixels. 0 means no solarization

            # Append view and view actions
            views[i] = self.normalize(self.to_tensor(view))
            actions.append(view_actions)

        return views, actions

if __name__ == "__main__":
    """
    Example usage:
      - Place an 'image.jpg' in the same directory, OR we'll synthesize a gradient placeholder.
      - It will generate:
          * 'episode_grid.png'  : grid of cropped views in order
          * 'episode_boxes.png' : original image with the sequence of crop boxes overlaid
    """
    import os

    NUM_VIEWS = 4
    transform = Episode_Transformations(
        num_views=NUM_VIEWS,
        size=224,
        geom_size=224,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        zoom_range=(0.08, 0.5),
        interpolation=transforms.InterpolationMode.BILINEAR
    )

    # Try to load 'image.jpg'
    path = "plots/testing_episode_creation/image.png"
    if os.path.exists(path):
        pil = Image.open(path).convert("RGB")
    else:
        # synthesize a smooth gradient image as fallback
        import numpy as np
        W, H = 640, 480
        x = np.linspace(0, 1, W)[None, :].repeat(H, axis=0)
        y = np.linspace(0, 1, H)[:, None].repeat(W, axis=1)
        arr = np.stack([x, y, 0.5 * (1 - x) + 0.5 * (1 - y)], axis=-1)
        arr = (255 * arr).astype("uint8")
        pil = Image.fromarray(arr, mode="RGB")
        pil.save("image.jpg")  # so user can see what was used

    # Generate episode
    views, actions = transform(pil)
    print(actions)

    # Prepare a grid for visualization (unnormalize for display)
    unnorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(transform.mean, transform.std)],
        std=[1.0 / s for s in transform.std],
    )
    disp_imgs = torch.stack([torch.clamp(unnorm(v), 0, 1) for v in views], dim=0)  # (V,3,H,W)

    # Save grid
    import torchvision
    grid = torchvision.utils.make_grid(disp_imgs, nrow=NUM_VIEWS)
    grid_img = transforms.ToPILImage()(grid.cpu())
    grid_img.save("plots/testing_episode_creation/episode_grid.png")
