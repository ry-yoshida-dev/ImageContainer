# Overview

Structural typing Protocol definitions for numpy-array-backed containers. Each protocol layers constraints used by mixins and the concrete ArrayImageContainer.

# Components

| Item | Description |
|------|-------------|
| [geometry.py](geometry.py) | Base geometry surface (value, channel_order). |
| [process.py](process.py) | Adds to_array for conversion-aware processing mixins. |
| [hash.py](hash.py) | Adds perceptual and related hash property accessors. |
