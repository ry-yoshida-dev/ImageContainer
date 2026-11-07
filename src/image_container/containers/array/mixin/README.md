# mixin

## Overview

Mixins that extend `ArrayImageContainer` with conversion, geometry, OpenCV processing and hashing, filters, and statistics.

## Components

| Component | Description |
|-----------|-------------|
| [convert.py](./convert.py) | `ArrayConvertMixin`: array, PIL, binary, and DCT exports; shared base for channel-order handling. |
| [geometry.py](./geometry.py) | `ArrayGeometryMixin`: shape and spatial accessors. |
| [hash.py](./hash.py) | `ArrayHashMixin`: perceptual hashes via OpenCV `img_hash`. |
| [filter.py](./filter.py) | `ArrayFilterMixin`: Laplacian and Sobel on a grayscale view. |
| [process.py](./process.py) | `ArrayProcessMixin`: histogram equalization, CLAHE, and LAB L-channel helpers. |
| [stats.py](./stats.py) | `ArrayStatsMixin`: gray/BGR means and histogram aggregates. |
