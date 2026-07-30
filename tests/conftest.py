"""Pytest defaults: never import mlx/Metal inside Cursor sandboxes.

``PP_MLX_ALLOW`` is the opt-in gate checked by ``mlx_runtime``; tests must never
set it. A bare ``import mlx.nn`` bypasses that gate entirely and SIGABRTs via
``metal::load_device``, so also reject ``mlx*`` at ``sys.meta_path`` — a stray
import then raises ImportError instead of crashing the interpreter.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Sequence

os.environ.pop("PP_MLX_ALLOW", None)
os.environ["PP_MLX_DISABLE"] = "1"


def _is_mlx_module(fullname: str) -> bool:
    root = fullname.split(".", 1)[0]
    return root == "mlx" or root.startswith("mlx_")


class BlockMlxLoader(importlib.abc.Loader):
    """Fail on execution so availability probes still work."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: object) -> None:
        name = getattr(module, "__name__", "mlx")
        raise ImportError(
            f"Refusing to import {name!r}: MLX is blocked under pytest. "
            "Metal is unavailable in sandboxed shells and the real import "
            "aborts the interpreter. Inject a stub via patch.dict(sys.modules) "
            "or pass a mock generate_fn."
        )


class BlockMlxImportFinder(importlib.abc.MetaPathFinder):
    """Claim every ``mlx*`` name so the real extension module never loads.

    ``find_spec`` must return a spec rather than raise: ``train._has_mlx()`` and
    similar probes call ``importlib.util.find_spec`` and legitimately expect an
    answer, not an exception. Execution is where Metal would abort, so the
    failure lands there instead.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not _is_mlx_module(fullname):
            return None
        spec = importlib.machinery.ModuleSpec(fullname, BlockMlxLoader())
        spec.submodule_search_locations = []
        return spec


for _cached in [name for name in sys.modules if _is_mlx_module(name)]:
    del sys.modules[_cached]

sys.meta_path[:] = [
    finder for finder in sys.meta_path if not isinstance(finder, BlockMlxImportFinder)
]
sys.meta_path.insert(0, BlockMlxImportFinder())
