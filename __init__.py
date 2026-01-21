"""
Phone Compression Ultimate - ComfyUI Custom Node
================================================

Simulates realistic smartphone camera compression and processing artifacts.

Installation:
    Copy this folder to ComfyUI/custom_nodes/
    Restart ComfyUI

Node Location:
    Add Node > image/postprocessing > 📱 Phone Compression Ultimate
"""

from .phone_compression_ultimate import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
