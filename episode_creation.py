import math
import random
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder
import hashlib

from PIL import ImageFilter, ImageOps, ImageFilter

class DeterministicEpisodes:
    """
    Wraps a stochastic Episode_Transformations to make its randomness
    deterministic per *image path* (key). This ensures a fixed val episode set.
    """
    def __init__(self, base_transform, base_seed: int = 0):
        self.base_transform = base_transform
        self.base_seed = int(base_seed)

    @staticmethod
    def _seed_from_key(key: str, base_seed: int) -> int:
        # Stable 64-bit hash → 31-bit torch seed
        h = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
        s = int.from_bytes(h, 'big') ^ base_seed
        return s % (2**31 - 1)

    def __call__(self, img, *, key: str):
        # Isolate RNG so we don't disturb global state
        py_state = random.getstate()
        try:
            seed = self._seed_from_key(key, self.base_seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                random.seed(seed)
                return self.base_transform(img)  # returns (views, aug_seq)
        finally:
            random.setstate(py_state)

class ImageFolderDetEpisodes(ImageFolder):
    """
    ImageFolder that passes the file path as a 'key' to DeterministicEpisodes.
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # Expect DeterministicEpisodes here
            sample = self.transform(sample, key=path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

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
        zoom_range: Tuple[float, float] = (0.0625, 0.5), #(0.0625, 0.25), #(0.2, 1.0), #(0.08, 0.8),
        interpolation=transforms.InterpolationMode.BILINEAR,
        p_crop: float = 1.0,
        p_hflip: float = 0.5,
        p_color_jitter: float = 0.8,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        p_threeaugment: float = 0.9,
        sigma_range: Tuple[float, float] = (0.1, 2.0)
    ):
        assert num_views > 0
        self.num_views = int(num_views)
        self.size = int(size)
        self.geom_size = int(geom_size)
        self.mean, self.std = mean, std
        self.zoom_range = (float(zoom_range[0]), float(zoom_range[1]))
        self.interp = interpolation

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        # crop
        self.p_crop = p_crop
        # horizontal flip
        self.p_hflip = p_hflip
        # color jitter
        self.p_color_jitter = p_color_jitter
        self.brightness_range = [1-brightness, 1+brightness]
        self.contrast_range = [1-contrast, 1+contrast]
        self.saturation_range = [1-saturation, 1+saturation]
        self.hue_range = [-hue, hue]
        # threeaugment
        self.p_threeaugment = p_threeaugment
        self.sigma_range = sigma_range
        
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

    @staticmethod
    def _minmax01(x, lo, hi):
        """map [lo,hi] → [0,1]"""
        return (x - lo) / (hi - lo)

    @staticmethod
    def _minmax11(x, lo, hi):
        """map [lo,hi] → [-1,1] (centre at 0)"""
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def _random_crop_image(self, base_img: Image.Image, p_crop: float, fix_center_crop: bool = False) -> Image.Image:
        crop_flag = False

        if torch.rand(1) < p_crop: # Apply crop
            crop_flag = True

            W, H = base_img.size  # should be 224,224
            R_hd = math.hypot(W / 2.0, H / 2.0) # Half-diagonal used for r normalization (constant for fixed W,H)

            if not fix_center_crop: # Sample random crop
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
                r_norm = _clamp(r_norm, 0.0, 1.0) # Clamp just in case of tiny FP overshoot
            else: # Get crop for the center of the image
                zoom = self._sample_zoom() #random.uniform(0.7, 1.0) #0.8 #self._sample_zoom()
                sin_th, cos_th = 0.0, 1.0
                side_px = _zoom_to_side_px(zoom, W, H)
                R = 0.0
                r_norm = 0.0

            # 5) Get top left h w from polar params and crop
            cx, cy = _center_from_R_px(sin_th, cos_th, R, W, H)
            top, left, h, w = _center_px_to_box(cx, cy, side_px, W, H)
            crop = F.resized_crop(base_img, top, left, h, w, [self.size, self.size], interpolation=self.interp)
            # 6) Create action tensor with polar params and zoom
            act = torch.tensor([sin_th, cos_th, r_norm, zoom], dtype=torch.float32)

        else: # no crop done, return the original image
            crop = base_img
            act = None

        return crop, act, crop_flag

    def _apply_random_flip(self, img, p_hflip):
        flip_code = 0
        hflip_flag=False
        if torch.rand(1) < p_hflip:
            img = F.hflip(img)
            flip_code = 1
            hflip_flag=True
        return img, torch.tensor([flip_code], dtype=torch.float), hflip_flag

    def _color_jitter(self, img_input, p_color_jitter, brightness_range, contrast_range, saturation_range, hue_range):
        # ColorJitter  -- use with caution. Values w/o _check_input for range test
        color_jitter_code = [1., 1., 1., 0.] # Values for no changes
        jitter_flag=False
        # Order: first brightness, then contrast, then saturation, then hue.
        if torch.rand(1) < p_color_jitter:
            brightness_factor = torch.empty(1).uniform_(brightness_range[0], brightness_range[1]).item()
            img = F.adjust_brightness(img_input, brightness_factor)
            color_jitter_code[0] = brightness_factor

            contrast_factor = torch.empty(1).uniform_(contrast_range[0], contrast_range[1]).item()
            img = F.adjust_contrast(img, contrast_factor)
            color_jitter_code[1] = contrast_factor

            saturation_factor = torch.empty(1).uniform_(saturation_range[0], saturation_range[1]).item()
            img = F.adjust_saturation(img, saturation_factor)
            color_jitter_code[2] = saturation_factor

            hue_factor = torch.empty(1).uniform_(hue_range[0], hue_range[1]).item()
            img = F.adjust_hue(img, hue_factor)
            color_jitter_code[3] = hue_factor
            jitter_flag=True
        else:
            img = img_input
        
        # Calculate the mean per channel of img_input and img. (img_input is a PIL image)
        # Idea from this paper: https://arxiv.org/abs/2306.06082
        img_input_tensor = self.totensor(img_input)
        mean_input = img_input_tensor.mean(dim=(1, 2))
        img_tensor = self.totensor(img)
        mean_output = img_tensor.mean(dim=(1, 2))
        # get mean diff
        mean_diff = mean_input - mean_output
        # Concatenate the mean diff with the color_jitter_code
        color_jitter_code = color_jitter_code + [mean_diff[0].item(), mean_diff[1].item(), mean_diff[2].item()]

        return img, torch.tensor(color_jitter_code, dtype=torch.float), jitter_flag

    def apply_random_grayscale(self, img, p_grayscale):
        # RandomGrayscale
        grayscale_code = 0
        gray_flag=False
        if torch.rand(1) < p_grayscale:
            num_output_channels, _, _ = F.get_dimensions(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            grayscale_code = 1
            gray_flag=True
        return img, torch.tensor([grayscale_code], dtype=torch.float), gray_flag

    def apply_gaussian_blur(self, img, p_gaussian_blur, sigma_range):
        # GaussianBlur
        gaussian_blur_code = 0.1 # applying 0.1 is basically no blur (output image values are the same as input image values)
        blur_flag=False
        if torch.rand(1) < p_gaussian_blur:
            sigma = torch.empty(1).uniform_(sigma_range[0], sigma_range[1]).item()
            img = img.filter(ImageFilter.GaussianBlur(sigma))
            gaussian_blur_code = sigma
            blur_flag=True
        return img, torch.tensor([gaussian_blur_code], dtype=torch.float), blur_flag
    
    def apply_solarization(self, img, p_solarization):
        # Solarization
        solarization_code = 0
        solar_flag=False
        if torch.rand(1) < p_solarization:
            # solarization_code = 1
            # solarization code is the ratio of pixel that are solarized to the total number of pixels
            solarization_code = torch.sum(self.totensor(img) > 0.5) / (3 * img.size[0] * img.size[1])
            img = ImageOps.solarize(img, threshold=128)
            solar_flag=True
        return img, torch.tensor([solarization_code], dtype=torch.float), solar_flag

    def __call__(self, pil_img: Image.Image):
        assert isinstance(pil_img, Image.Image), "Input must be a PIL.Image"
        base_img = self._ensure_geom_image(pil_img)  # enforce 224×224 geometry

        views = torch.zeros(self.num_views, 3, base_img.size[0], base_img.size[1])
        actions = []

        for i in range(self.num_views):
            view_actions = []

            ## Crop
            if i ==0: # First view (crop is always at the center of the image. Any size)
                view, action_cropbb, crop_flag = self._random_crop_image(base_img, self.p_crop, fix_center_crop = True)
                if crop_flag:
                    view_actions.append(("crop", action_cropbb))
                # No other augs for the first view
                views[i] = self.normalize(self.totensor(view))
                actions.append(view_actions)
                continue
            else: # other views: random crop
                view, action_cropbb, crop_flag = self._random_crop_image(base_img, self.p_crop)
                if crop_flag:
                    view_actions.append(("crop", action_cropbb))

            # ## Horizontal Flip 
            # view, action_horizontalflip, hflip_flag = self._apply_random_flip(view, self.p_hflip)
            # if hflip_flag:
            #     view_actions.append(("hflip", torch.empty(0))) # param-less. Only needs type_emb later on.

            ## ColorJitter
            view, action_colorjitter, jitter_flag = self._color_jitter(view, p_color_jitter = self.p_color_jitter, 
                                                                    brightness_range = self.brightness_range, 
                                                                    contrast_range = self.contrast_range, 
                                                                    saturation_range = self.saturation_range, 
                                                                    hue_range = self.hue_range)
            if jitter_flag:
                action_colorjitter[0] = self._minmax11(action_colorjitter[0], self.brightness_range[0], self.brightness_range[1])
                action_colorjitter[1] = self._minmax11(action_colorjitter[1], self.contrast_range[0], self.contrast_range[1])
                action_colorjitter[2] = self._minmax11(action_colorjitter[2], self.saturation_range[0], self.saturation_range[1])
                action_colorjitter[3] = self._minmax11(action_colorjitter[3], self.hue_range[0], self.hue_range[1])
                view_actions.append(("jitter", action_colorjitter))

            if torch.rand(1) < self.p_threeaugment: # apply threeaugment with a probability
                # ThreeAgument from Deit3: Randomly choose between grayscale, gaussian blur, and solarization (uniformly)
                choice = random.randint(0, 2)
                ## Grayscale
                if choice == 0:
                    view, action_grayscale, gray_flag = self.apply_random_grayscale(view, p_grayscale = 1.0)
                    if gray_flag:
                        view_actions.append(("gray", torch.empty(0))) # param-less. Only needs type_emb later on.
                ## Gaussian Blur
                elif choice == 1:
                    view, action_gaussianblur, blur_flag = self.apply_gaussian_blur(view, p_gaussian_blur = 1.0, sigma_range = self.sigma_range)
                    if blur_flag:
                        sigma_log  = math.log(action_gaussianblur.item() / self.sigma_range[0]) / math.log(self.sigma_range[1] / self.sigma_range[0])
                        sigma_norm = 2.0 * sigma_log - 1.0 # [-1, 1]
                        view_actions.append(("blur", torch.tensor([sigma_norm], dtype=torch.float)))
                ## Solarization
                elif choice == 2:
                    view, action_solarization, solar_flag = self.apply_solarization(view, p_solarization = 1.0)
                    if solar_flag:
                        view_actions.append(("solar", action_solarization))

            # Append view and view actions
            views[i] = self.normalize(self.totensor(view))
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
    transform = Episode_Transformations(num_views=NUM_VIEWS)

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
    for act in actions:
        print(act)

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
