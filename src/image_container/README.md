# image_container

## Overview

A unified container class designed to maintain compatibility across different image formats including PIL images, BGR/RGB arrays, and other image representations.

## Components

| Component | Description |
|-----------|-------------|
| [ch_order.py](./ch_order.py) | Defines channel-order enums such as RGB and BGR. |
| [format.py](./format.py) | Defines the image-format enum used across the package. |
| [extention.py](./extention.py) | Defines supported file-extension enums for path collection. |
| [iterator.py](./iterator.py) | Provides directory-based image iteration and format-specific container creation. |
| [container.py](./container.py) | Defines the base container and the factory-style registration entry point. |
| [containers/](./containers/) | Contains concrete container implementations for each image format. |
| [binary_image/](./binary_image/) | Provides binary-image helpers and the dedicated binary-image container. |
| [dct_image/](./dct_image/) | Provides DCT coefficient image container, IDCT restoration, and frequency-domain Laplacian. |
| [utils/](./utils/) | Shared helpers such as spatial padding for even-sized processing. |

## Examples

### Register In-Memory Images

```python
import numpy as np
from PIL import Image

from image_container.ch_order import ChannelOrder
from image_container.container import ImageContainer

# NumPy array input (BGR)
array_image = np.zeros((256, 256, 3), dtype=np.uint8)
array_container = ImageContainer.register(array_image, ChannelOrder.BGR)
array_resized = array_container.resize(width=128, height=128)

# PIL input (RGB)
pil_image = Image.new("RGB", (256, 256), color=(0, 0, 0))
pil_container = ImageContainer.register(pil_image, ChannelOrder.RGB)
pil_resized = pil_container.resize(width=128, height=128)
```

### Iterate Images From Directory

```python
from image_container.format import ImageFormat
from image_container.iterator import ImageIterator

# Directory iteration (supports HEIC via pillow-heif opener registration)
image_iterator = ImageIterator.from_dir(
    dir_path="path/to/images",
    image_format=ImageFormat.ARRAY,
)
for image_container in image_iterator:
    resized_container = image_container.resize(width=128, height=128)
```