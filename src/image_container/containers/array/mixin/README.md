# Overview

Mixins that extend array-backed containers with conversion, layout-aware geometry, OpenCV processing and hashing (opencv-contrib img_hash), and aggregate statistics. ArrayConvertMixin is the shared base for methods that need to_array and channel-order handling.

# Components

| Item | Description |
|------|-------------|
| [convert.py](convert.py) | ArrayConvertMixin: to_array, PIL/binary/DCT exports, grayscale and 3-channel helpers. |
| [geometry.py](geometry.py) | ArrayGeometryMixin: shape and spatial accessors over value and channel_order. |
| [hash.py](hash.py) | ArrayHashMixin: perceptual and related hashes via OpenCV img_hash on BGR views. |
| [filter.py](filter.py) | ArrayFilterMixin: Laplacian and Sobel on a grayscale view via OpenCV. |
| [process.py](process.py) | ArrayProcessMixin: histogram equalization, CLAHE, and LAB L-channel helpers. |
| [stats.py](stats.py) | ArrayStatsMixin: gray/BGR means and histogram-related aggregates. |
