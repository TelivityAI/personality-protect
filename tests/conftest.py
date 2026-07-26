"""Pytest defaults: never import mlx/Metal inside Cursor sandboxes."""

from __future__ import annotations

import os

# mlx.core metal::load_device SIGABRTs in sandboxed shells (~80ms). Block it.
os.environ.setdefault("PP_MLX_DISABLE", "1")
