# 🚀 QUICK START GUIDE

## Installation (30 seconds)

1. **Download** `phone_compression_ultimate.py`
2. **Copy** to your ComfyUI folder:
   ```
   ComfyUI/custom_nodes/phone_compression_ultimate.py
   ```
3. **Restart** ComfyUI
4. **Find node** at: `Add Node > image > postprocessing > Phone Compression Ultimate 📱`

Done! ✅

---

## First Use (1 minute)

### Basic Workflow:

```
[Load Image] 
    ↓
[Phone Compression Ultimate]
    ↓
[Preview Image]
```

### Settings:
1. **Preset:** `iPhone_15_Pro` (or any other)
2. **Global Intensity:** `1.0` (realistic) or `0.5` (subtle)
3. Press **Queue Prompt**

That's it! 🎉

---

## Presets Explained (30 seconds)

| Preset | When to Use |
|--------|-------------|
| **iPhone_15_Pro** | Clean, balanced look - portraits, general use |
| **Samsung_S24_Ultra** | Vibrant, sharp - landscapes, colorful scenes |
| **Pixel_9_Pro** | High contrast - night mode, HDR scenes |
| **Generic_DayLight** | Subtle - when you want minimal artifacts |
| **Generic_LowLight** | Heavy artifacts - indoor, night, security cam style |

---

## Common Scenarios

### "I want it to look like an iPhone photo"
```
Preset: iPhone_15_Pro
Global Intensity: 0.8-1.0
```

### "I want exaggerated Samsung over-sharpening"
```
Preset: Samsung_S24_Ultra
Global Intensity: 1.2-1.5
```

### "I want security camera / found footage style"
```
Preset: Generic_LowLight
Global Intensity: 1.5
```

### "I want to remove AI 'too clean' look"
```
Preset: Generic_DayLight
Global Intensity: 0.5-0.7
```

---

## Video Processing

### Simple Video Workflow:

```
[VHS Video Decode]
    ↓
[Phone Compression Ultimate]
    ├─ Preset: iPhone_15_Pro
    ├─ Global Intensity: 0.8
    └─ Temporal Consistency: 0.8
    ↓
[VHS Video Encode]
```

**Important:** Set `temporal_consistency` to 0.7-0.9 for smooth video!

---

## Customization

### Want to tweak individual effects?

1. Select `preset: Custom_Mode`
2. All sliders become active
3. Adjust any parameter
4. Values -1 = disabled, 0-1 = enabled

### Save your custom settings:

1. Configure all parameters
2. Type name in `save_preset_name` (e.g., "my_style")
3. Process image
4. Restart ComfyUI
5. Your preset now appears in dropdown!

---

## Troubleshooting

**Node not showing?**
- Check file is in `custom_nodes/` folder
- Restart ComfyUI

**Effect too strong?**
- Lower `global_intensity` to 0.5

**Effect too weak?**
- Increase `global_intensity` to 1.5

**Video flickering?**
- Increase `temporal_consistency` to 0.9

---

## Tips & Tricks

💡 **For AI images:** Use intensity 0.6-0.8 (removes "too perfect" look)  
💡 **For upscaled images:** Use intensity 1.0-1.2 (adds back realistic artifacts)  
💡 **For video:** ALWAYS enable temporal_consistency (0.75-0.85)  
💡 **For Instagram aesthetic:** iPhone preset + slight boost to saturation  
💡 **For vintage phone:** Lower JPEG quality (70-80) + high noise reduction  

---

## Next Steps

- Read full **README.md** for detailed parameter guide
- Experiment with different presets
- Create your own custom presets
- Combine with other ComfyUI nodes

**Have fun! 📱✨**
