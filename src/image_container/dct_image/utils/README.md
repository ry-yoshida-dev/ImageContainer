# utils

## Overview

Internal helpers for block-wise DCT and zigzag coefficient ordering used by BlockedDctImage. Not part of the public dct_image API.

## Components

| Component | Description |
|-----------|-------------|
| [block_transform.py](./block_transform.py) | Block-wise forward/inverse DCT via row-wise cv2.dct / cv2.idct over all blocks at once. |
| [zigzag_scan.py](./zigzag_scan.py) | JPEG zigzag scan order and conversion between zigzag coefficient vectors and the block layout. |
