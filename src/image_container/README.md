# image_container

## Overview

A unified container class that keeps PIL images, BGR/RGB arrays, and torch tensors interchangeable.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Defines NumPy array type aliases: `UInt8Image` for uint8 pixel buffers, `ImageArray` for general processing arrays, and `BinaryArray` for masks. |
| [ch_order.py](./ch_order.py) | Defines channel-order enums such as RGB and BGR. |
| [format.py](./format.py) | Defines the image-format enum used across the package. |
| [extention.py](./extention.py) | Defines supported file-extension enums for path collection. |
| [iterator.py](./iterator.py) | Provides directory-based image iteration and format-specific container creation. |
| [container.py](./container.py) | Defines the base container and the factory-style registration entry point. |
| [containers/](./containers/README.md) | Contains concrete container implementations for each image format. |
| [binary_image/](./binary_image/README.md) | Provides binary-image helpers and the dedicated binary-image container. |
| [dct_image/](./dct_image/README.md) | Provides NonBlockedDctImage, BlockedDctImage, and ZigzagDctCoefficients containers, IDCT restoration, and frequency-domain Laplacian. |
| [utils/](./utils/README.md) | Shared helpers such as spatial padding for even-sized processing. |

## Examples

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
