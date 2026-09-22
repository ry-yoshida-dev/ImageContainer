# dct_image

## Overview

Utilities for working with 2D DCT coefficient images, including global and block-wise construction with spatial padding via [utils/padding.py](../utils/padding.py), conversion from JPEG zigzag-ordered (optionally truncated) block coefficients, inverse DCT restoration, radial low/high-pass filtering in the frequency domain, and frequency-domain Laplacian edge emphasis.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Shared DctImageBase abstract container with validation, IDCT, DC/AC coefficient views, radial filters, and Laplacian helpers. |
| [container/](./container/) | NonBlockedDctImage, BlockedDctImage, and ZigzagDctCoefficients concrete containers. |
| [utils/](./utils/) | Internal block-wise DCT transform and zigzag scan helpers (not exported). |

## Examples

```python
from image_container import ZigzagDctCoefficients

zigzag = ZigzagDctCoefficients(value=coefficients, block_size=8)
image = zigzag.to_blocked()
restored = image.idct(is_uint8_cast_enabled=False)
```

`coefficients` is a float32 array of shape `(blocks_y, blocks_x, coefficient_count)` in JPEG zigzag order. Missing high-frequency slots are zero, so a truncated vector restores a low-pass approximation, and `image.coefficient_count` records how many slots carry data.
