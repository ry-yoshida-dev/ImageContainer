# dct_image

## Overview

Utilities for working with 2D DCT coefficient images, including global and block-wise construction with spatial padding via [utils/padding.py](../utils/padding.py), inverse DCT restoration, radial low/high-pass filtering in the frequency domain, and frequency-domain Laplacian edge emphasis.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Shared DctImageBase abstract container with validation, IDCT, DC/AC coefficient views, radial filters, and Laplacian helpers. |
| [container/](./container/) | NonBlockedDctImage and BlockedDctImage concrete containers. |
| [utils/](./utils/) | Internal block-wise DCT matrix cache and transform helpers (not exported). |
