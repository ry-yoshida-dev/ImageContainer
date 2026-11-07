# array

## Overview

NumPy-backed `ArrayImageContainer`, composed of mixins for conversion, geometry, hashing, statistics, OpenCV processing, and derivative filters.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `ArrayImageContainer`: validates the uint8 payload and wires the mixins. |
| [mixin/](./mixin/README.md) | Behavior split into focused mixins. |
| [protocols/](./protocols/README.md) | Layered `Protocol` types for static typing of the mixins. |

## Examples

```python
import numpy as np

from image_container.ch_order import ChannelOrder
from image_container.containers.array.container import ArrayImageContainer

image = np.zeros((64, 64, 3), dtype=np.uint8)
container = ArrayImageContainer(value=image, channel_order=ChannelOrder.BGR)

gray = container.to_gray()                  # shape (64, 64)
equalized = container.equalize_histogram()  # shape (64, 64, 3)
phash = container.phash                     # shape (1, 8), dtype uint8
```
