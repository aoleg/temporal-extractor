"""
temporal_extractor -- extract a small number of high-quality stills from a
low-quality video, reconstructing each still from a window of neighbouring
frames rather than from a single frame.

Nothing at package level imports the restorer, so importing this package never
drags in torch. See restore/ for why that separation exists.
"""

__version__ = "0.1.0"
