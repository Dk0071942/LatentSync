"""
Standardized FFmpeg encoding configuration for LatentSync project.

This module provides consistent encoding parameters across all video processing operations.
Based on the project requirements for high-quality video output with broad compatibility.
"""

from typing import Dict, List

# Standard video encoding parameters for high-quality output
STANDARD_VIDEO_ENCODING = {
    "codec": "libx264",
    "preset": "slow",
    "crf": "18",
    "pix_fmt": "yuv420p",
    "color_primaries": "bt709",
    "color_trc": "bt709", 
    "colorspace": "bt709",
    "movflags": "+faststart",
    "vf": "format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1"
}

# Standard audio encoding parameters
STANDARD_AUDIO_ENCODING = {
    "codec": "aac",
    "bitrate": "192k",
    "sample_rate": "16000"
}

# Legacy/fast encoding for intermediate processing
FAST_VIDEO_ENCODING = {
    "codec": "libx264", 
    "preset": "veryfast",
    "crf": "18",
    "pix_fmt": "yuv420p",
    "movflags": "+faststart"
}

def get_standard_video_params() -> List[str]:
    """
    Get standard video encoding parameters as ffmpeg command line arguments.
    
    Returns:
        List of ffmpeg parameters for high-quality video encoding
    """
    return [
        "-c:v", STANDARD_VIDEO_ENCODING["codec"],
        "-preset", STANDARD_VIDEO_ENCODING["preset"], 
        "-crf", STANDARD_VIDEO_ENCODING["crf"],
        "-pix_fmt", STANDARD_VIDEO_ENCODING["pix_fmt"],
        "-vf", STANDARD_VIDEO_ENCODING["vf"],
        "-color_primaries", STANDARD_VIDEO_ENCODING["color_primaries"],
        "-color_trc", STANDARD_VIDEO_ENCODING["color_trc"],
        "-colorspace", STANDARD_VIDEO_ENCODING["colorspace"],
        "-movflags", STANDARD_VIDEO_ENCODING["movflags"]
    ]

def get_standard_audio_params() -> List[str]:
    """
    Get standard audio encoding parameters as ffmpeg command line arguments.
    
    Returns:
        List of ffmpeg parameters for audio encoding
    """
    return [
        "-c:a", STANDARD_AUDIO_ENCODING["codec"],
        "-b:a", STANDARD_AUDIO_ENCODING["bitrate"],
        "-ar", STANDARD_AUDIO_ENCODING["sample_rate"]
    ]

def get_fast_video_params() -> List[str]:
    """
    Get fast video encoding parameters for intermediate processing.
    
    Returns:
        List of ffmpeg parameters for fast video encoding
    """
    return [
        "-c:v", FAST_VIDEO_ENCODING["codec"],
        "-preset", FAST_VIDEO_ENCODING["preset"],
        "-crf", FAST_VIDEO_ENCODING["crf"], 
        "-pix_fmt", FAST_VIDEO_ENCODING["pix_fmt"],
        "-movflags", FAST_VIDEO_ENCODING["movflags"]
    ]

def get_imageio_params() -> List[str]:
    """
    Get encoding parameters compatible with imageio-ffmpeg for write_video function.
    
    Returns:
        List of ffmpeg parameters for imageio compatibility
    """
    return [
        "-preset", STANDARD_VIDEO_ENCODING["preset"],
        "-crf", STANDARD_VIDEO_ENCODING["crf"],
        "-pix_fmt", STANDARD_VIDEO_ENCODING["pix_fmt"]
    ]

def build_standard_command(input_file: str, output_file: str, use_fast: bool = False) -> List[str]:
    """
    Build a complete ffmpeg command with standard encoding parameters.
    
    Args:
        input_file: Path to input video file
        output_file: Path to output video file  
        use_fast: If True, use fast encoding preset for intermediate processing
        
    Returns:
        Complete ffmpeg command as list of strings
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-i", input_file]
    
    if use_fast:
        cmd.extend(get_fast_video_params())
    else:
        cmd.extend(get_standard_video_params())
    
    cmd.extend(get_standard_audio_params())
    cmd.append(output_file)
    
    return cmd