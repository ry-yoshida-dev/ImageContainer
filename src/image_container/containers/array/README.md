# Overview

NumPy-backed image container implementation: ArrayImageContainer composes mixins for conversion, geometry, hashing, statistics, and OpenCV-oriented processing. Structural protocols live under [protocols/](protocols/).

# Components

| Item | Description |
|------|-------------|
| [container.py](container.py) | Concrete ArrayImageContainer; validates payload, exposes ImageFormat.ARRAY, wires mixins and SupportsArrayHash. |
| [mixin/](mixin/) | Reusable behavior split into focused mixins (see [mixin/README.md](mixin/README.md)). |
| [protocols/](protocols/) | Layered Protocol types for static typing (SupportsArrayGeometry -> process -> hash). |

# Usage/Examples

```python
import numpy as np

from image_container.ch_order import ChannelOrder
from image_container.containers.array.container import ArrayImageContainer

image = np.zeros((64, 64, 3), dtype=np.uint8)
container = ArrayImageContainer(value=image, channel_order=ChannelOrder.BGR)

gray = container.to_gray()                  # convert: shape (64, 64)
shape = container.shape                     # geometry: (64, 64, 3)
mean = container.mean_gray                  # stats: float
equalized = container.equalize_histogram()  # process: shape (64, 64, 3)
phash = container.phash                     # hash: shape (1, 8), dtype uint8
```
