import os
import numpy as np
import cv2
from PIL import Image

def resize_im(w, h, scale=416, max_scale=608):
    f = float(scale) / min(h, w)
    if max_scale is not None:
        if f * max(h, w) > max_scale:
            f = float(max_scale) / max(h, w)
    newW, newH = int(w * f), int(h * f)
    return newW - (newW % 32), newH - (newH % 32)