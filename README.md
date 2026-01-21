# Phone Compression Ultimate 📱

A ComfyUI custom node that simulates realistic smartphone camera compression and processing artifacts.

## Features

- **5 Built-in Presets**: iPhone 15 Pro, Samsung S24 Ultra, Pixel 9 Pro, Generic DayLight, Generic LowLight
- **7 Independent Processing Modules** (can be toggled on/off):
  - Sensor Noise (ISO-dependent luminance and chroma noise)
  - JPEG Compression (DCT blocking, mosquito noise, color banding)
  - ISP Processing (noise reduction, over-sharpening, HDR tone mapping)
  - Shadow Detail Loss (posterization, pixelation, noise crawl, detail crushing)
  - Color Science (brand-specific color shifts, saturation boost)
  - Lens Aberrations (chromatic aberration, vignetting)
  - Temporal Consistency (for video - reduces flickering between frames)
- **Custom Preset System**: Save and load your own presets as JSON files
- **Full Parameter Control**: Override any preset value with custom settings
- **GPU Accelerated**: Uses PyTorch for efficient processing

## Installation

1. Copy the `comfyui_phone_compression_ultimate` folder to your `ComfyUI/custom_nodes/` directory
2. Restart ComfyUI
3. The node appears in: **Add Node → image/postprocessing → Phone Compression Ultimate 📱**

## Usage

### Basic Usage
1. Connect an image to the node's input
2. Select a preset (e.g., "iPhone_15_Pro")
3. Adjust `global_intensity` to control overall effect strength (0.0-2.0)

### Custom Mode
1. Set preset to "Custom_Mode"
2. Adjust individual parameters (values ≥ 0 override preset values)
3. Parameters set to -1 use the preset's default values

### Saving Custom Presets
1. Configure your desired settings
2. Enter a name in `save_preset_name`
3. Run the node - preset will be saved to `~/.comfyui_phone_presets/`
4. Your preset will appear in the preset dropdown on next restart

### Video Processing
1. Enable `temporal_consistency` (0.0-1.0) to reduce flickering between frames
2. Use `reset_temporal` to clear the frame buffer when starting a new video

## Parameter Reference

### Module Toggles
- `enable_sensor_noise`: Enable/disable sensor noise simulation
- `enable_jpeg`: Enable/disable JPEG compression artifacts
- `enable_isp`: Enable/disable ISP processing effects
- `enable_shadow_loss`: Enable/disable shadow detail loss
- `enable_color`: Enable/disable color science
- `enable_lens`: Enable/disable lens aberrations

### Sensor Noise Module
| Parameter | Range | Description |
|-----------|-------|-------------|
| `iso_level` | 100-6400 | Simulated ISO setting (higher = more noise) |
| `luminance_noise` | 0-1 | Grayscale grain intensity |
| `chroma_noise` | 0-1 | Color speckle intensity |

### JPEG Compression Module
| Parameter | Range | Description |
|-----------|-------|-------------|
| `jpeg_quality` | 60-100 | JPEG quality (lower = more compression artifacts) |
| `mosquito_noise` | 0-1 | Edge ringing/noise intensity |
| `color_banding` | 0-1 | Gradient posterization intensity |

### ISP Processing Module
| Parameter | Range | Description |
|-----------|-------|-------------|
| `noise_reduction` | 0-1 | Aggressive smoothing (watercolor effect) |
| `sharpening` | 0-2 | Over-sharpening with halo artifacts |
| `hdr_processing` | 0-1 | HDR tone mapping artifacts |

### Shadow Detail Loss Module ⭐
| Parameter | Range | Description |
|-----------|-------|-------------|
| `shadow_posterization` | 0-1 | Blocky shadows (color banding in dark areas) |
| `shadow_pixelation` | 0-1 | Macroblocking in shadows |
| `shadow_noise_crawl` | 0-1 | Dancing grain in dark areas |
| `detail_crushing` | 0-1 | Loss of texture in deepest shadows |

### Color Science Module
| Parameter | Range | Description |
|-----------|-------|-------------|
| `color_shift` | preset/neutral/warm/cool/green/cool_magenta | Brand-specific color cast |
| `saturation_boost` | 0.8-1.5 | Color saturation multiplier |

### Lens Aberrations Module
| Parameter | Range | Description |
|-----------|-------|-------------|
| `chromatic_aberration` | 0-1 | Color fringing at edges |
| `vignetting` | 0-1 | Darkened corners |

### Temporal Module
| Parameter | Range | Description |
|-----------|-------|-------------|
| `temporal_consistency` | 0-1 | Frame blending for video (reduces flicker) |
| `reset_temporal` | boolean | Clear frame buffer |

## Preset Characteristics

### iPhone 15 Pro
- Warm color cast
- Balanced processing
- Moderate sharpening
- Good shadow detail preservation
- Natural-looking output

### Samsung S24 Ultra
- Green tint in certain lighting
- **Aggressive over-sharpening** (most extreme)
- Over-saturated colors
- Strong shadow artifacts
- "Instagram-ready" look

### Pixel 9 Pro
- Cool/magenta tint
- High contrast
- Strong HDR processing
- Good detail preservation
- "Computational photography" look

### Generic DayLight
- Minimal artifacts
- Neutral colors
- Light processing
- Good for subtle effects

### Generic LowLight
- Heavy sensor noise
- Strong shadow posterization
- Visible pixelation in shadows
- "Crawling pixels" effect
- Simulates challenging low-light conditions

## Tips

1. **Start with a preset** and adjust `global_intensity` first
2. **Shadow Detail Loss** is the most important module for realism
3. **Use low values** (0.2-0.5) for subtle effects
4. **Disable modules** you don't need for faster processing
5. **Save presets** for consistent results across projects

## Requirements

- ComfyUI
- PyTorch (comes with ComfyUI)

## License

MIT License - see LICENSE file for details.

## Version History

See CHANGELOG.md for full version history.

### v1.1.0 (Current)
- Fixed: `__init__.py` now properly imports the node
- Fixed: ISP noise reduction edge-preserving blend
- Fixed: JPEG compression block artifacts
- Fixed: Shadow detail loss posterization mixing
- Fixed: Temporal consistency frame buffer management
- Fixed: Chromatic aberration edge handling
- Added: `reset_temporal` parameter for video processing
- Improved: Better dtype handling (float32 processing)
- Improved: More realistic shadow region detection
- Improved: Smoother vignette falloff

### v1.0.0
- Initial release
