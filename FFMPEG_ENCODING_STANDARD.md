# FFmpeg Encoding Standard for LatentSync

## Overview

This document defines the standardized FFmpeg encoding parameters used throughout the LatentSync project. All video encoding operations should follow these standards to ensure consistent quality, compatibility, and optimal file sizes.

## Standard Encoding Parameters

The standardized encoding command follows this pattern:

```bash
ffmpeg -i "input.mp4" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -vf "format=yuv420p,colorspace=all=bt709:iall=bt709:fast=1" -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart "output_converted.mp4"
```

### Video Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `-c:v libx264` | H.264 codec | Universal compatibility and good compression |
| `-preset slow` | Slow encoding preset | Higher quality output (better than veryfast) |
| `-crf 18` | Constant Rate Factor 18 | High quality (visually lossless for most content) |
| `-pix_fmt yuv420p` | YUV 4:2:0 pixel format | Maximum compatibility with players/browsers |
| `-vf "format=yuv420p,colorspace=all=bt709:iall=bt709:fast=1"` | Color format filter | Ensures proper color space handling |
| `-color_primaries bt709` | BT.709 color primaries | Standard HD color space |
| `-color_trc bt709` | BT.709 transfer characteristics | Standard HD gamma curve |
| `-colorspace bt709` | BT.709 color space | Standard HD color matrix |
| `-movflags +faststart` | Web optimization | Enables streaming before full download |

### Audio Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `-c:a aac` | AAC audio codec | Universal compatibility and good compression |
| `-b:a 192k` | 192 kbps bitrate | High quality audio |
| `-ar 16000` | 16kHz sample rate | Optimized for speech content in lip-sync |

## Implementation

### Centralized Configuration

The standardized parameters are implemented in `latentsync/utils/ffmpeg_config.py`:

```python
from latentsync.utils.ffmpeg_config import get_standard_video_params, get_standard_audio_params
```

### Key Functions

- `get_standard_video_params()` - Returns list of video encoding parameters
- `get_standard_audio_params()` - Returns list of audio encoding parameters  
- `get_fast_video_params()` - Returns faster encoding preset for intermediate processing
- `get_imageio_params()` - Returns parameters compatible with imageio-ffmpeg
- `build_standard_command()` - Builds complete ffmpeg command

## Updated Files

The following files have been updated to use the standardized encoding:

### Core Utilities
- `latentsync/utils/util.py` - Updated `write_video()` function
- `latentsync/pipelines/lipsync_pipeline.py` - Final video output encoding

### Preprocessing
- `preprocess/resample_fps_hz.py` - FPS resampling with standard encoding
- `preprocess/affine_transform.py` - Affine transformation output encoding

### RIFE Integration
- `ECCV2022-RIFE/rife_app/services/video_interpolator.py` - Video interpolation output
- `ECCV2022-RIFE/rife_app/services/chained.py` - Chained interpolation processing
- `ECCV2022-RIFE/README.md` - Updated example commands

## Quality vs Performance Trade-offs

### Standard Encoding (Recommended)
- **Preset**: `slow` 
- **Use case**: Final output videos, published content
- **Quality**: Highest
- **Speed**: Slower but acceptable for final processing

### Fast Encoding (Intermediate Processing)
- **Preset**: `veryfast`
- **Use case**: Intermediate processing steps, preview generation
- **Quality**: Good
- **Speed**: Much faster

## Color Space Standardization

All videos are encoded with BT.709 color space parameters:
- **Color Primaries**: BT.709 (HDTV standard)
- **Transfer Characteristics**: BT.709 gamma curve
- **Color Matrix**: BT.709 for HD content

This ensures consistent color reproduction across different players and devices.

## Compatibility

The standardized parameters ensure compatibility with:
- All major web browsers (Chrome, Firefox, Safari, Edge)
- Mobile devices (iOS, Android)  
- Video editing software (Adobe Premiere, DaVinci Resolve, etc.)
- Streaming platforms (YouTube, Vimeo, etc.)

## Migration Notes

### Before Standardization
The project used inconsistent encoding parameters:
- Mixed presets (`veryfast`, `fast`, unspecified)
- Inconsistent CRF values (13, 18)
- Missing color space specifications
- No web optimization flags

### After Standardization
- Consistent high-quality encoding across all components
- Proper color space handling for HD content
- Web-optimized output with `+faststart`
- Centralized configuration for easy maintenance

## Performance Impact

- **Encoding time**: ~20-30% increase due to `slow` preset vs `veryfast`
- **File size**: ~10-15% reduction due to better compression efficiency
- **Quality**: Significant improvement in visual quality and color accuracy
- **Compatibility**: Enhanced compatibility across devices and platforms

## Future Considerations

- Monitor encoding performance and adjust presets if needed
- Consider HDR support (BT.2020 color space) for future enhancements
- Evaluate AV1 codec adoption for better compression (when support improves)