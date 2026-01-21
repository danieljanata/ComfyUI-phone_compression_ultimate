# Changelog

All notable changes to Phone Compression Ultimate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-18

### Added
- Initial release
- 7 independent processing modules:
  - Sensor Noise Module (ISO-dependent luminance & chroma noise)
  - JPEG Compression Module (DCT blocks, mosquito noise, color banding)
  - ISP Processing Module (noise reduction, sharpening, HDR)
  - Shadow Detail Loss Module (posterization, pixelation, noise crawl, detail crushing)
  - Color Science Module (brand-specific color shifts & saturation)
  - Lens Aberrations Module (chromatic aberration, vignetting)
  - Temporal Module (frame consistency for video)

- 5 built-in presets:
  - iPhone 15 Pro (warm, balanced)
  - Samsung S24 Ultra (extreme sharpening, oversaturated)
  - Pixel 9 Pro (high contrast, cool/magenta tint)
  - Generic DayLight (subtle, clean)
  - Generic LowLight (heavy artifacts, high noise)

- Custom preset system:
  - Save custom presets to JSON
  - Load saved presets
  - Automatic preset directory management (~/.comfyui_phone_presets)

- Video support:
  - Temporal consistency (prevents flickering)
  - Batch processing optimization
  - Frame-to-frame blending

- GPU acceleration:
  - PyTorch CUDA support
  - Automatic device detection
  - Mixed precision support

### Features
- Global intensity master control (0.0 - 2.0)
- Individual module toggles (enable/disable any module)
- Per-parameter override system (-1 = use preset, 0+ = custom value)
- Real-time preset switching
- Professional ISP pipeline simulation
- Brand-specific processing signatures

### Performance
- Optimized for batch processing
- GPU-accelerated operations
- Efficient memory usage
- ~50-200ms per frame (depends on resolution & GPU)

### Documentation
- Complete README.md with parameter guide
- Quick Start Guide (QUICKSTART.md)
- Technical documentation
- Example use cases
- Troubleshooting guide

---

## [Unreleased]

### Planned Features (Future Versions)
- [ ] Additional phone presets (OnePlus, Xiaomi, Oppo)
- [ ] Night mode specific artifacts
- [ ] Portrait mode edge detection errors
- [ ] Ultrawide lens distortion
- [ ] Beauty mode smoothing
- [ ] RAW → JPEG pipeline simulation
- [ ] Per-frame noise pattern variation
- [ ] Advanced temporal noise (video-specific)
- [ ] Lens flare simulation
- [ ] Motion blur artifacts
- [ ] Auto ISO adjustment based on image brightness

### Potential Improvements
- [ ] Performance optimization for 4K+ resolution
- [ ] Preset import/export (.zip bundles)
- [ ] Preset preview thumbnails
- [ ] Real-time preview mode
- [ ] Batch preset application
- [ ] Parameter randomization
- [ ] A/B comparison view
- [ ] Metadata embedding (which preset was used)

---

## Version History Summary

- **v1.0.0** (2026-01-18) - Initial public release

---

## Contributing

If you'd like to contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Ideas
- New phone brand presets
- Performance optimizations
- Bug fixes
- Documentation improvements
- Example workflows
- Test cases

---

## Bug Reports

Please include:
- ComfyUI version
- Input image characteristics
- Settings/preset used
- Expected vs actual behavior
- Console error messages (if any)

---

## Credits

**Development:** Daniel Janata & Claude (Anthropic)  
**Research:** Real-world smartphone ISP pipeline analysis  
**Testing:** iPhone 15 Pro, Samsung S24 Ultra, Pixel 9 Pro

---

**Last Updated:** 2026-01-18
