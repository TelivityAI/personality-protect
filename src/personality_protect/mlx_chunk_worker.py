"""Subprocess entrypoint for one memory-capped MLX LoRA chunk.

Caps ``mx.set_wired_limit`` before mlx-lm's trainer can request ~40 GB on a
48 GB Mac (which jetsam-kills Python and freezes the machine).
"""

from __future__ import annotations

import argparse
import os
import sys
import types

from personality_protect.mlx_runtime import install_wired_cap


def _patch_empty_val_dataset() -> None:
    """mlx-lm wraps empty valid sets in CacheDataset (truthy) and then div-by-zero evals."""
    from mlx_lm.tuner import trainer as trainer_mod

    original = trainer_mod.train

    def _train(*args, **kwargs):  # noqa: ANN002, ANN003
        val = kwargs.get("val_dataset")
        if val is not None and len(val) == 0:
            kwargs["val_dataset"] = None
        if args and len(args) >= 4:
            # positional: model, optimizer, train_dataset, val_dataset, ...
            pass
        return original(*args, **kwargs)

    trainer_mod.train = _train  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PersonalityProtect MLX chunk worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--wired-bytes", type=int, required=True)
    parser.add_argument("--resume-adapter-file", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args(argv)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["PP_MLX_WIRED_BYTES"] = str(args.wired_bytes)

    install_wired_cap(args.wired_bytes)
    _patch_empty_val_dataset()

    from mlx_lm.lora import CONFIG_DEFAULTS, run

    # Build args namespace matching mlx_lm.lora expectations
    ns = types.SimpleNamespace(**dict(CONFIG_DEFAULTS))
    ns.model = args.model
    ns.data = args.data
    ns.train = True
    ns.test = False
    ns.batch_size = max(1, args.batch_size)
    ns.iters = max(1, args.iters)
    ns.adapter_path = args.adapter_path
    ns.max_seq_length = args.max_seq_length
    ns.num_layers = args.num_layers
    ns.grad_checkpoint = True
    ns.steps_per_report = 5
    ns.steps_per_eval = 10**9
    ns.val_batches = 0
    ns.save_every = max(1, args.iters)
    ns.learning_rate = args.learning_rate
    ns.resume_adapter_file = args.resume_adapter_file
    ns.fine_tune_type = "lora"
    ns.report_to = None
    ns.clear_cache_threshold = 2 * 10**9  # clear allocator if cache > 2 GB

    print(
        f"PP MLX chunk: iters={ns.iters} wired_cap_gb={args.wired_bytes / 1e9:.1f} "
        f"max_seq={ns.max_seq_length} layers={ns.num_layers}",
        flush=True,
    )
    run(ns)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"PP MLX chunk failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
