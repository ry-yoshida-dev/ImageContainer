# image_container

## Overview

A unified container class designed to maintain compatibility across different image formats including PIL images, BGR/RGB arrays, and other image representations.

## Components

| Component | Description |
|-----------|-------------|
| [ch_order.py](./ch_order.py) | Defines channel order specifications (e.g., RGB, BGR). |
| [format.py](./format.py) | Defines type union for supported image formats. |
| [container.py](./container.py) | Base class and ImageContainer.register() for registering images to container classes. |
| [containers/](./containers/) | Container implementations per image format. |
| [binary_image/](./binary_image/) | Binary image utilities and container class. |