# containers

## Overview

Concrete `ImageContainer` implementations, one per supported image format, together with the `ImageContainerType` union that names every container `ImageContainer.register` can return.

## Components

| Component | Description |
|-----------|-------------|
| [array/](./array/README.md) | `ArrayImageContainer` for NumPy uint8 pixel buffers. |
| [pil/](./pil/README.md) | `PILImageContainer` for `PIL.Image.Image` images. |
| [tensor/](./tensor/README.md) | `TensorImageContainer` for `torch.Tensor` images kept on their own device. |
