# pil

## Overview

`PILImageContainer` for `PIL.Image.Image` images, composed of conversion, geometry, and statistics mixins.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `PILImageContainer`: validates the PIL payload and wires the mixins. |
| [protocol.py](./protocol.py) | `SupportsPILGeometry` structural type used by the geometry mixin. |
| [mixin/](./mixin/README.md) | Behavior split into focused mixins. |
