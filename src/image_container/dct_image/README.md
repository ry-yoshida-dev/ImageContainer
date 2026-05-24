# dct_image

## Overview

Utilities for working with 2D DCT coefficient images, including array-based construction with odd-dimension padding via [utils/padding.py](../utils/padding.py), inverse DCT restoration, and frequency-domain Laplacian edge emphasis.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | DCT image container with validation, construction, IDCT restoration, and frequency-domain Laplacian. |
