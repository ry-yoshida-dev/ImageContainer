# tests

## Overview

Pytest suite for `image_container`.

## Components

| Component | Description |
|-----------|-------------|
| [conftest.py](./conftest.py) | Shared fixtures (random generator, sample images). |
| [test_image_container.py](./test_image_container.py) | `ImageContainer.register` and base container behavior. |
| [test_array_container.py](./test_array_container.py) | `ArrayImageContainer`. |
| [test_pil_container.py](./test_pil_container.py) | `PILImageContainer`. |
| [test_tensor_container.py](./test_tensor_container.py) | `TensorImageContainer`. |
| [test_binary_image.py](./test_binary_image.py) | `BinaryImage` and connected components. |
| [test_dct_image.py](./test_dct_image.py) | DCT coefficient containers. |
| [test_ch_order.py](./test_ch_order.py) | `ChannelOrder`. |
| [test_format.py](./test_format.py) | `ImageFormat`. |
| [test_extention.py](./test_extention.py) | File-extension enums. |
| [test_iterator.py](./test_iterator.py) | `ImageIterator`. |
| [test_padding.py](./test_padding.py) | Padding utilities. |
