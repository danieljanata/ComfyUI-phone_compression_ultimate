"""
Phone Compression Ultimate - ComfyUI Custom Node
================================================

Simulates realistic smartphone camera compression and processing artifacts.
Combines REAL JPEG compression with adaptive shadow processing.

Features:
- Real JPEG compression via PIL
- Adaptive noise (more in dark areas, like real sensors)
- Shadow detail loss with posterization and RANDOM blocking
- Detail smudging (watercolor effect from noise reduction)
- Brand-specific color science
- Instagram/social media compression simulation

Author: Claude & Daniel Janata
Version: 4.0.0
License: MIT
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
from typing import Tuple


# ============================================================================
# PRESETS - Realistic phone camera settings
# ============================================================================

PRESETS = {
    "iPhone_15_Pro": {
        "jpeg_quality": 78,
        "noise_amount": 0.015,
        "noise_in_shadows": 0.05,
        "sharpening": 1.3,
        "saturation": 1.08,
        "shadow_posterize": 0.25,
        "shadow_block_size": 4,
        "block_randomness": 0.4,          # NEW: Random block variation
        "detail_smudge": 0.3,             # NEW: Watercolor/smudge effect
        "color_temp": "warm",
        "vignette": 0.15,
        "highlight_compression": 0.1,
    },
    "Samsung_S24_Ultra": {
        "jpeg_quality": 70,
        "noise_amount": 0.02,
        "noise_in_shadows": 0.07,
        "sharpening": 1.9,
        "saturation": 1.3,
        "shadow_posterize": 0.35,
        "shadow_block_size": 5,
        "block_randomness": 0.5,
        "detail_smudge": 0.5,             # Samsung has aggressive noise reduction
        "color_temp": "green",
        "vignette": 0.2,
        "highlight_compression": 0.15,
    },
    "Pixel_9_Pro": {
        "jpeg_quality": 75,
        "noise_amount": 0.018,
        "noise_in_shadows": 0.06,
        "sharpening": 1.4,
        "saturation": 1.05,
        "shadow_posterize": 0.2,
        "shadow_block_size": 4,
        "block_randomness": 0.35,
        "detail_smudge": 0.4,
        "color_temp": "cool",
        "vignette": 0.12,
        "highlight_compression": 0.2,
    },
    "Budget_Android": {
        "jpeg_quality": 55,
        "noise_amount": 0.04,
        "noise_in_shadows": 0.12,
        "sharpening": 2.0,
        "saturation": 1.35,
        "shadow_posterize": 0.5,
        "shadow_block_size": 8,
        "block_randomness": 0.7,
        "detail_smudge": 0.7,             # Heavy noise reduction = more smudge
        "color_temp": "warm",
        "vignette": 0.3,
        "highlight_compression": 0.05,
    },
    "Low_Light_Indoor": {
        "jpeg_quality": 60,
        "noise_amount": 0.05,
        "noise_in_shadows": 0.15,
        "sharpening": 1.6,
        "saturation": 1.15,
        "shadow_posterize": 0.6,
        "shadow_block_size": 10,
        "block_randomness": 0.8,
        "detail_smudge": 0.8,             # Very smudgy in low light
        "color_temp": "warm",
        "vignette": 0.25,
        "highlight_compression": 0.25,
    },
    "Instagram_Repost": {
        "jpeg_quality": 65,
        "noise_amount": 0.025,
        "noise_in_shadows": 0.08,
        "sharpening": 1.5,
        "saturation": 1.2,
        "shadow_posterize": 0.4,
        "shadow_block_size": 6,
        "block_randomness": 0.6,
        "detail_smudge": 0.55,
        "color_temp": "neutral",
        "vignette": 0.1,
        "highlight_compression": 0.1,
    },
    "WhatsApp_Compressed": {
        "jpeg_quality": 50,
        "noise_amount": 0.035,
        "noise_in_shadows": 0.1,
        "sharpening": 1.3,
        "saturation": 1.1,
        "shadow_posterize": 0.55,
        "shadow_block_size": 8,
        "block_randomness": 0.75,
        "detail_smudge": 0.65,
        "color_temp": "neutral",
        "vignette": 0.05,
        "highlight_compression": 0.05,
    },
    "Selfie_Front_Camera": {
        "jpeg_quality": 72,
        "noise_amount": 0.03,
        "noise_in_shadows": 0.09,
        "sharpening": 1.2,
        "saturation": 1.15,
        "shadow_posterize": 0.3,
        "shadow_block_size": 5,
        "block_randomness": 0.5,
        "detail_smudge": 0.6,             # Beauty mode = smudgy skin
        "color_temp": "warm",
        "vignette": 0.1,
        "highlight_compression": 0.15,
    },
}


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI tensor [B,H,W,C] to numpy [H,W,C] float 0-1."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.cpu().numpy()


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy [H,W,C] to ComfyUI tensor [1,H,W,C]."""
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert numpy float 0-1 to PIL Image."""
    return Image.fromarray((array * 255).clip(0, 255).astype(np.uint8), 'RGB')


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy float 0-1."""
    return np.array(image).astype(np.float32) / 255.0


def get_brightness_mask(image: np.ndarray) -> np.ndarray:
    """Create mask: dark areas = 1, bright areas = 0."""
    brightness = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    shadow_mask = np.clip((0.35 - brightness) / 0.35, 0, 1)
    return shadow_mask


def get_deep_shadow_mask(image: np.ndarray) -> np.ndarray:
    """Mask for very dark areas (< 0.15 brightness)."""
    brightness = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    return np.clip((0.15 - brightness) / 0.15, 0, 1)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply real JPEG compression using PIL."""
    pil_img = numpy_to_pil(image)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    
    compressed = Image.open(buffer).convert('RGB')
    return pil_to_numpy(compressed)


def apply_adaptive_noise(image: np.ndarray, base_noise: float, shadow_noise: float, 
                         intensity: float = 1.0) -> np.ndarray:
    """Add noise that's stronger in dark areas."""
    if base_noise <= 0 and shadow_noise <= 0:
        return image
    
    shadow_mask = get_brightness_mask(image)
    shadow_mask_3d = np.stack([shadow_mask] * 3, axis=2)
    
    # Base noise
    base = np.random.randn(*image.shape) * base_noise * intensity
    
    # Extra noise in shadows
    shadow = np.random.randn(*image.shape) * shadow_noise * intensity * shadow_mask_3d
    
    result = image + base + shadow
    return np.clip(result, 0, 1)


def apply_detail_smudge(image: np.ndarray, amount: float, 
                        intensity: float = 1.0) -> np.ndarray:
    """
    Apply "smudge" effect - simulates aggressive noise reduction.
    This creates the "washed out" / "watercolor" look typical of phone cameras.
    Fine details get blurred together while edges are somewhat preserved.
    """
    if amount <= 0:
        return image
    
    amount *= intensity
    h, w, c = image.shape
    
    pil_img = numpy_to_pil(image)
    
    # Multi-scale blur for organic smudge effect
    # Small blur for fine detail loss
    blur_small = pil_img.filter(ImageFilter.GaussianBlur(radius=1.0 + amount * 1.5))
    # Medium blur for larger area smoothing
    blur_medium = pil_img.filter(ImageFilter.GaussianBlur(radius=2.0 + amount * 2.5))
    
    img_np = image.copy()
    blur_small_np = pil_to_numpy(blur_small)
    blur_medium_np = pil_to_numpy(blur_medium)
    
    # Edge detection for edge-preserving blur
    # We want to blur smooth areas more, preserve edges
    gray = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    
    # Sobel-like edge detection
    gy, gx = np.gradient(gray)
    edge_strength = np.sqrt(gx**2 + gy**2)
    
    # Normalize and create edge mask (edges = 1, smooth = 0)
    edge_mask = np.clip(edge_strength * 15, 0, 1)
    
    # Invert: smooth areas = 1 (will be blurred), edges = 0 (preserved)
    smooth_mask = 1 - edge_mask
    smooth_mask_3d = np.stack([smooth_mask] * 3, axis=2)
    
    # Blend: smooth areas get blurred, edges preserved
    # More blur in shadow areas too
    shadow_mask = get_brightness_mask(image)[:, :, np.newaxis]
    
    # Mix of blurs based on area type
    blur_mix = blur_small_np * 0.6 + blur_medium_np * 0.4
    
    # Apply more smudge in smooth areas and shadows
    blend_strength = smooth_mask_3d * amount * 0.7 + shadow_mask * amount * 0.4
    blend_strength = np.clip(blend_strength, 0, 0.85)
    
    result = img_np * (1 - blend_strength) + blur_mix * blend_strength
    
    return np.clip(result, 0, 1)


def apply_shadow_posterization(image: np.ndarray, amount: float, 
                               intensity: float = 1.0) -> np.ndarray:
    """Reduce color depth in shadow areas."""
    if amount <= 0:
        return image
    
    amount *= intensity
    shadow_mask = get_brightness_mask(image)
    deep_shadow_mask = get_deep_shadow_mask(image)
    
    # Different levels for different shadow depths
    deep_levels = max(4, int(16 * (1 - amount)))
    mid_levels = max(8, int(32 * (1 - amount * 0.7)))
    
    deep_posterized = np.floor(image * deep_levels) / deep_levels
    mid_posterized = np.floor(image * mid_levels) / mid_levels
    
    result = image.copy()
    
    # Mid shadows
    mid_mask = shadow_mask[:, :, np.newaxis] * amount * 0.6
    result = result * (1 - mid_mask) + mid_posterized * mid_mask
    
    # Deep shadows
    deep_mask = deep_shadow_mask[:, :, np.newaxis] * amount * 0.9
    result = result * (1 - deep_mask) + deep_posterized * deep_mask
    
    return result


def apply_random_block_artifacts(image: np.ndarray, block_size: int, 
                                  randomness: float, amount: float,
                                  intensity: float = 1.0) -> np.ndarray:
    """
    Create IRREGULAR/RANDOM block artifacts (not grid-aligned).
    This looks more like real compression artifacts.
    """
    if block_size < 2 or amount <= 0 or randomness <= 0:
        return image
    
    amount *= intensity
    randomness *= intensity
    h, w, c = image.shape
    
    shadow_mask = get_brightness_mask(image)
    
    # Create irregular regions using noise-based approach
    # Generate random "region map" 
    region_scale = max(2, block_size)
    
    # Create low-frequency noise for region boundaries
    small_h = max(4, h // region_scale)
    small_w = max(4, w // region_scale)
    
    # Random values for each small region
    region_noise = np.random.rand(small_h, small_w)
    
    # Add some variation with different frequencies
    region_noise2 = np.random.rand(small_h * 2, small_w * 2)
    
    # Resize to full image size
    pil_noise = Image.fromarray((region_noise * 255).astype(np.uint8), 'L')
    pil_noise = pil_noise.resize((w, h), Image.Resampling.NEAREST)
    region_map = np.array(pil_noise).astype(np.float32) / 255.0
    
    pil_noise2 = Image.fromarray((region_noise2 * 255).astype(np.uint8), 'L')
    pil_noise2 = pil_noise2.resize((w, h), Image.Resampling.BILINEAR)
    region_map2 = np.array(pil_noise2).astype(np.float32) / 255.0
    
    # Combine for more organic look
    region_map = region_map * 0.6 + region_map2 * 0.4
    
    # Quantize region map to create distinct regions
    num_regions = max(4, 16 - int(randomness * 10))
    region_map = np.floor(region_map * num_regions) / num_regions
    
    # For each "region", average the colors (creates blocky look)
    result = image.copy()
    
    # Create pooled version at different scales
    pil_img = numpy_to_pil(image)
    
    # Multiple block sizes for variety
    block_sizes = [
        max(2, block_size - 2),
        block_size,
        max(2, block_size + 2)
    ]
    
    pooled_versions = []
    for bs in block_sizes:
        small = pil_img.resize((max(1, w // bs), max(1, h // bs)), Image.Resampling.BILINEAR)
        large = small.resize((w, h), Image.Resampling.NEAREST)
        pooled_versions.append(pil_to_numpy(large))
    
    # Use region map to select which pooled version to use
    for i, threshold in enumerate([0.33, 0.66, 1.0]):
        if i < len(pooled_versions):
            mask = ((region_map >= (threshold - 0.33)) & (region_map < threshold)).astype(np.float32)
            mask = mask[:, :, np.newaxis]
            
            # Apply more in shadows
            apply_mask = mask * shadow_mask[:, :, np.newaxis] * amount * randomness
            result = result * (1 - apply_mask) + pooled_versions[i] * apply_mask
    
    # Add some random color shifts per region (very subtle)
    color_shift = (np.random.rand(small_h, small_w, 3) - 0.5) * 0.03 * randomness * amount
    pil_shift = Image.fromarray(((color_shift + 0.5) * 255).clip(0, 255).astype(np.uint8), 'RGB')
    pil_shift = pil_shift.resize((w, h), Image.Resampling.NEAREST)
    color_shift_full = (np.array(pil_shift).astype(np.float32) / 255.0 - 0.5) * 2
    
    # Apply color shift only in shadows
    shadow_3d = shadow_mask[:, :, np.newaxis]
    result = result + color_shift_full * shadow_3d * amount * 0.3
    
    return np.clip(result, 0, 1)


def apply_sharpening(image: np.ndarray, amount: float, 
                     intensity: float = 1.0) -> np.ndarray:
    """Apply sharpening with halos."""
    if amount <= 1.0:
        return image
    
    amount = 1.0 + (amount - 1.0) * intensity
    
    pil_img = numpy_to_pil(image)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1.2))
    
    img_np = pil_to_numpy(pil_img)
    blur_np = pil_to_numpy(blurred)
    
    # Unsharp mask
    detail = img_np - blur_np
    sharpened = img_np + detail * (amount - 1.0) * 2.5
    
    return np.clip(sharpened, 0, 1)


def apply_color_temperature(image: np.ndarray, temp: str, 
                            intensity: float = 1.0) -> np.ndarray:
    """Apply color temperature shift."""
    if temp == "neutral":
        return image
    
    result = image.copy()
    
    if temp == "warm":
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 + 0.04 * intensity), 0, 1)
        result[:, :, 1] = np.clip(result[:, :, 1] * (1 + 0.01 * intensity), 0, 1)
        result[:, :, 2] = np.clip(result[:, :, 2] * (1 - 0.04 * intensity), 0, 1)
    elif temp == "cool":
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 - 0.03 * intensity), 0, 1)
        result[:, :, 2] = np.clip(result[:, :, 2] * (1 + 0.04 * intensity), 0, 1)
    elif temp == "green":
        result[:, :, 1] = np.clip(result[:, :, 1] * (1 + 0.03 * intensity), 0, 1)
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 - 0.01 * intensity), 0, 1)
    
    return result


def apply_saturation(image: np.ndarray, amount: float, 
                     intensity: float = 1.0) -> np.ndarray:
    """Adjust color saturation."""
    if amount == 1.0:
        return image
    
    amount = 1.0 + (amount - 1.0) * intensity
    
    gray = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    gray_3d = np.stack([gray] * 3, axis=2)
    
    result = gray_3d + (image - gray_3d) * amount
    
    return np.clip(result, 0, 1)


def apply_vignette(image: np.ndarray, amount: float, 
                   intensity: float = 1.0) -> np.ndarray:
    """Apply vignette (dark corners)."""
    if amount <= 0:
        return image
    
    amount *= intensity
    h, w = image.shape[:2]
    
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    
    dist = np.sqrt((x - center_x)**2 / (center_x**2) + (y - center_y)**2 / (center_y**2))
    
    vignette = 1 - np.clip((dist - 0.7) * amount * 1.5, 0, 0.5)
    vignette_3d = vignette[:, :, np.newaxis]
    
    return image * vignette_3d


def apply_highlight_compression(image: np.ndarray, amount: float,
                                intensity: float = 1.0) -> np.ndarray:
    """HDR-like effect."""
    if amount <= 0:
        return image
    
    amount *= intensity
    
    brightness = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    
    # Lift shadows
    shadow_mask = np.clip((0.3 - brightness) / 0.3, 0, 1)[:, :, np.newaxis]
    lifted = image + shadow_mask * amount * 0.15
    
    # Compress highlights
    highlight_mask = np.clip((brightness - 0.7) / 0.3, 0, 1)[:, :, np.newaxis]
    compressed = lifted - highlight_mask * (lifted - 0.85) * amount * 0.5
    
    return np.clip(compressed, 0, 1)


# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class PhoneCompressionUltimate:
    """
    Simulates smartphone camera compression and processing.
    
    NEW in v4.0:
    - Random block artifacts (not grid-aligned)
    - Detail smudge effect (watercolor/washed look)
    - More realistic presets
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (list(PRESETS.keys()) + ["Custom"], {
                    "default": "iPhone_15_Pro",
                    "tooltip": "Phone camera preset. Each has unique processing characteristics."
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Overall effect strength. 1.0 = realistic, 2.0 = exaggerated"
                }),
            },
            "optional": {
                # === COMPRESSION ===
                "jpeg_quality": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "step": 5,
                    "tooltip": "JPEG quality (1-100). Lower = more artifacts. -1 = use preset"
                }),
                
                # === NOISE ===
                "noise_amount": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 0.15,
                    "step": 0.005,
                    "display": "slider",
                    "tooltip": "Base sensor noise. -1 = use preset"
                }),
                "noise_in_shadows": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 0.3,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "EXTRA noise in dark areas. Key for realism! -1 = use preset"
                }),
                
                # === DETAIL SMUDGE (NEW!) ===
                "detail_smudge": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Detail smudging / watercolor effect. Simulates aggressive noise reduction. Higher = more washed out look. -1 = use preset"
                }),
                
                # === SHADOW ARTIFACTS ===
                "shadow_posterize": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Color banding in shadows. -1 = use preset"
                }),
                "shadow_block_size": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Base block size in shadows (2-16). -1 = use preset"
                }),
                "block_randomness": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Block irregularity. 0 = grid-like, 1 = very random/organic blocks. -1 = use preset"
                }),
                
                # === ISP PROCESSING ===
                "sharpening": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Sharpening. >1.5 = halos. -1 = use preset"
                }),
                "saturation": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Color saturation. -1 = use preset"
                }),
                
                # === COLOR ===
                "color_temp": (["preset", "neutral", "warm", "cool", "green"], {
                    "default": "preset",
                    "tooltip": "Color temperature. warm=iPhone, green=Samsung, cool=Pixel"
                }),
                
                # === LENS ===
                "vignette": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Dark corners. -1 = use preset"
                }),
                
                # === HDR ===
                "highlight_compression": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "HDR effect. -1 = use preset"
                }),
                
                # === EXTRA ===
                "double_compression": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply JPEG twice (screenshot reupload simulation)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"
    
    DESCRIPTION = """📱 Phone Compression Ultimate v4.0

REALISTIC smartphone camera artifacts.

NEW FEATURES:
• detail_smudge - "Washed out" / watercolor effect from noise reduction
• block_randomness - Irregular blocks (not grid-aligned)

KEY EFFECTS:
• JPEG compression - Real, not simulated
• Adaptive noise - More grain in dark areas
• Shadow posterization - Banding in dark areas
• Random blocking - Organic artifact patterns
• Detail smudge - Fine detail loss (watercolor)

PRESETS:
• iPhone_15_Pro - Warm, balanced
• Samsung_S24_Ultra - Green, sharp, vivid, smudgy
• Pixel_9_Pro - Cool, HDR
• Budget_Android - Heavy artifacts
• Low_Light_Indoor - Noisy, smudgy
• Instagram_Repost - Social media look
• WhatsApp_Compressed - Heavy compression
• Selfie_Front_Camera - Beauty mode smudge

TIP: Use intensity 1.0-1.5 for realistic results."""

    def process(
        self,
        image: torch.Tensor,
        preset: str,
        intensity: float = 1.0,
        jpeg_quality: int = -1,
        noise_amount: float = -1.0,
        noise_in_shadows: float = -1.0,
        detail_smudge: float = -1.0,
        shadow_posterize: float = -1.0,
        shadow_block_size: int = -1,
        block_randomness: float = -1.0,
        sharpening: float = -1.0,
        saturation: float = -1.0,
        color_temp: str = "preset",
        vignette: float = -1.0,
        highlight_compression: float = -1.0,
        double_compression: bool = False,
    ):
        if intensity <= 0:
            return (image,)
        
        # Get preset
        p = PRESETS.get(preset, PRESETS["iPhone_15_Pro"])
        
        # Resolve parameters
        final_jpeg = jpeg_quality if jpeg_quality >= 0 else p["jpeg_quality"]
        final_noise = noise_amount if noise_amount >= 0 else p["noise_amount"]
        final_shadow_noise = noise_in_shadows if noise_in_shadows >= 0 else p["noise_in_shadows"]
        final_smudge = detail_smudge if detail_smudge >= 0 else p["detail_smudge"]
        final_posterize = shadow_posterize if shadow_posterize >= 0 else p["shadow_posterize"]
        final_block = shadow_block_size if shadow_block_size >= 0 else p["shadow_block_size"]
        final_randomness = block_randomness if block_randomness >= 0 else p["block_randomness"]
        final_sharp = sharpening if sharpening >= 0 else p["sharpening"]
        final_sat = saturation if saturation >= 0 else p["saturation"]
        final_temp = color_temp if color_temp != "preset" else p["color_temp"]
        final_vignette = vignette if vignette >= 0 else p["vignette"]
        final_hdr = highlight_compression if highlight_compression >= 0 else p["highlight_compression"]
        
        # Adjust JPEG quality based on intensity
        if intensity > 1.0:
            final_jpeg = max(20, int(final_jpeg - (intensity - 1.0) * 25))
        
        # Process each image in batch
        results = []
        
        for i in range(image.shape[0]):
            img = tensor_to_numpy(image[i:i+1])
            
            # === PROCESSING PIPELINE ===
            
            # 1. HDR / Highlight compression
            if final_hdr > 0:
                img = apply_highlight_compression(img, final_hdr, intensity)
            
            # 2. Color temperature
            img = apply_color_temperature(img, final_temp, intensity)
            
            # 3. Saturation
            img = apply_saturation(img, final_sat, intensity)
            
            # 4. Sharpening
            img = apply_sharpening(img, final_sharp, intensity)
            
            # 5. Detail smudge (noise reduction simulation) - BEFORE noise!
            if final_smudge > 0:
                img = apply_detail_smudge(img, final_smudge, intensity)
            
            # 6. Adaptive noise
            img = apply_adaptive_noise(img, final_noise, final_shadow_noise, intensity)
            
            # 7. Shadow posterization
            img = apply_shadow_posterization(img, final_posterize, intensity)
            
            # 8. Random block artifacts
            if final_randomness > 0:
                img = apply_random_block_artifacts(img, final_block, final_randomness, 
                                                   final_posterize, intensity)
            
            # 9. Vignette
            img = apply_vignette(img, final_vignette, intensity)
            
            # 10. JPEG compression
            img = apply_jpeg_compression(img, final_jpeg)
            
            # 11. Double compression
            if double_compression:
                img = apply_adaptive_noise(img, 0.008, 0.012, 1.0)
                img = apply_detail_smudge(img, 0.2, 1.0)
                img = apply_jpeg_compression(img, max(40, final_jpeg - 15))
            
            results.append(numpy_to_tensor(img))
        
        result = torch.cat(results, dim=0)
        
        return (result,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PhoneCompressionUltimate": PhoneCompressionUltimate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhoneCompressionUltimate": "📱 Phone Compression Ultimate"
}
