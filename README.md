# ImageContainer

## Overview

`image_container` provides a unified interface for image data held as NumPy arrays, PIL images, or torch tensors, with explicit channel order handling (RGB, BGR, GRAY). It also offers binary-image analysis, DCT coefficient images, and directory-based image iteration.

## Installation

```bash
pip install .
```

## Usage

```python
from PIL import Image

from image_container import ChannelOrder, ImageContainer

image = ImageContainer.register(
    image=Image.open("image.png"),
    channel_order=ChannelOrder.RGB,
)

array_bgr = image.to_array(ch_order=ChannelOrder.BGR)
cropped = image.crop((slice(50, 200), slice(100, 300)))
cropped.save("out/cropped.png")
```
