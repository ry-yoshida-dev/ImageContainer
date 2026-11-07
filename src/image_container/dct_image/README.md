# dct_image

## Overview

Containers for 2D DCT coefficient images: global and block-wise transforms, JPEG zigzag-ordered (optionally truncated) block coefficients, inverse DCT restoration, and frequency-domain filtering.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Shared DctImageBase abstract container with validation, IDCT, DC/AC coefficient views, radial filters, and Laplacian helpers. |
| [container/](./container/README.md) | NonBlockedDctImage, BlockedDctImage, and ZigzagDctCoefficients concrete containers. |
| [utils/](./utils/README.md) | Internal block-wise DCT transform and zigzag scan helpers (not exported). |

## Examples

```python
from image_container import ZigzagDctCoefficients

zigzag = ZigzagDctCoefficients(value=coefficients, block_size=8)
image = zigzag.to_blocked()
restored = image.idct(is_uint8_cast_enabled=False)
```

