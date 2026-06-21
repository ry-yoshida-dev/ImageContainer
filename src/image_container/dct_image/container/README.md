# container

## Overview

Concrete DCT coefficient image containers for global and block-wise transforms.

## Components

| Component | Description |
|-----------|-------------|
| [non_blocked.py](./non_blocked.py) | NonBlockedDctImage for global forward/inverse DCT (filters inherited from base). |
| [blocked.py](./blocked.py) | BlockedDctImage for block-wise forward/inverse DCT and per-block radial masks. |
