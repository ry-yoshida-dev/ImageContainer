# container

## Overview

Concrete DCT coefficient containers for global and block-wise transforms, and for per-block zigzag-ordered coefficient vectors. The coefficient layout is fixed by the container type.

## Components

| Component | Description |
|-----------|-------------|
| [non_blocked.py](./non_blocked.py) | NonBlockedDctImage for global forward/inverse DCT (filters inherited from base). |
| [blocked.py](./blocked.py) | BlockedDctImage for block-wise forward/inverse DCT in spatial (row-major per block) layout, with the count of available zigzag coefficients. |
| [zigzag.py](./zigzag.py) | ZigzagDctCoefficients holding per-block JPEG zigzag vectors, optionally truncated, and converting to and from BlockedDctImage. |
