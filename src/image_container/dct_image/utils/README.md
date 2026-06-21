# utils

## Overview

Internal helpers for block-wise DCT construction used by BlockedDctImage. Not part of the public dct_image API.

## Components

| Component | Description |
|-----------|-------------|
| [block_transform.py](./block_transform.py) | Vectorized block-wise forward/inverse DCT via batched matrix multiplication. |
| [dct_matrix.py](./dct_matrix.py) | LRU-cached OpenCV-compatible 2D DCT matrices built via batched row-wise cv2.dct / cv2.idct. |
