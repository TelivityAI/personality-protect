"""PersonalityProtect CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from personality_protect import __version__
from personality_protect.api import DEFAULT_HOST, DEFAULT_PORT, serve as serve_api
from personality_protect.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_SIZE_HINT,
    DEFAULT_MIN_WORDS,
    DEFAULT_MLX_SIZE_HINT,
    DEFAULT_PROFILE,
    DEFAULT_THROUGH_YEAR,
    default_home,
    get_paths,
    init_profile,
    load_config,
)
from personality_protect.demo import run_demo
from personality_protect.download import run_download
from personality_protect.filter import filter_draft, read_draft_input
from personality_protect.ingest import run_ingest
from personality_protect.logo import (
    ColorMode,
    LogoDisplay,
    LogoMode,
    print_logo,
    should_show_logo,
)
from personality_protect.models import load_index, summarize_by_source_year
from personality_protect.select import run_select
from personality_protect.train import backend_docs, detect_backend, run_train

app = typer.Typer(
    name="personality-protect",
    help="Local-only personal writing-voice filter (Telivity).",
    no_args_is_help=False,
    add_completion=False,
)
console = Console()


def _show_banner(
    color: ColorMode,
    logo: LogoDisplay,
    logo_mode: LogoMode,
    *,
    machine_readable: bool = False,
) -> None:
    if logo == "off" or not should_show_logo(machine_readable=machine_readable):
        return
    print_logo(logo_mode, color=color, display=logo)


def _banner_from_ctx(ctx: typer.Context, *, json_mode: bool = False) -> None:
    _show_banner(
        ctx.obj.get("color", "auto"),
        ctx.obj.get("logo", "full"),
        ctx.obj.get("logo_mode", "auto"),
        machine_readable=json_mode,
    )


def _print_summary(summary: dict, title: str = "Selection") -> None:
    console.print(f"[bold]{title}[/bold]: {summary.get('pieces', 0)} pieces, "
                  f"{summary.get('words', 0)} words")
    table = Table(title="By source")
    table.add_column("Source")
    table.add_column("Count", justify="right")
    for src, count in (summary.get("by_source") or {}).items():
        table.add_row(src, str(count))
    console.print(table)
    table2 = Table(title="By year")
    table2.add_column("Year")
    table2.add_column("Count", justify="right")
    for year, count in (summary.get("by_year") or {}).items():
        table2.add_row(year, str(count))
    console.print(table2)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
    color: ColorMode = typer.Option(
        "auto",
        "--color",
        help="Color output: auto | always | never.",
    ),
    logo: LogoDisplay = typer.Option(
        "full",
        "--logo",
        help="Logo display: full | mark | off.",
    ),
    logo_mode: LogoMode = typer.Option(
        "auto",
        "--logo-mode",
        help="Logo render mode: auto | truecolor | color | plain | ascii.",
    ),
) -> None:
    """PersonalityProtect — your writing, your weights, your machine."""
    ctx.ensure_object(dict)
    ctx.obj["color"] = color
    ctx.obj["logo"] = logo
    ctx.obj["logo_mode"] = logo_mode

    if version:
        typer.echo(f"personality-protect {__version__}")
        raise typer.Exit(0)

    if ctx.invoked_subcommand is None:
        _show_banner(color, logo, logo_mode)
        typer.echo("PersonalityProtect — local voice filter")
        typer.echo("Run with --help for commands. Data never leaves this machine.")
        typer.echo("")
        typer.echo(
            "Commands: init | download | ingest | select | train | filter | demo | api | logo"
        )


@app.command("logo")
def logo_cmd(
    ctx: typer.Context,
    mode: Optional[LogoMode] = typer.Option(
        None,
        "--mode",
        help="Override render mode for this command.",
    ),
    mark_only: bool = typer.Option(
        False,
        "--mark-only",
        help="Omit the Telivity wordmark.",
    ),
) -> None:
    """Print the Telivity terminal logo (for verification)."""
    color: ColorMode = ctx.obj.get("color", "auto")
    logo_mode: LogoMode = mode or ctx.obj.get("logo_mode", "auto")
    display: LogoDisplay = "mark" if mark_only else "full"
    print_logo(logo_mode, mark_only=mark_only, color=color, display=display)


@app.command("init")
def init_cmd(
    ctx: typer.Context,
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile", help="Profile name."),
    home: Optional[Path] = typer.Option(
        None,
        "--home",
        help="Override state dir (default: ~/.personality-protect).",
    ),
    base_model: str = typer.Option(
        DEFAULT_BASE_MODEL,
        "--base-model",
        help="Quantized train/filter model id (default: MLX 4-bit ~6 GB).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite config."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Create a local profile directory (corpus/adapters stay here)."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths, config, created = init_profile(
        profile, home=home, base_model=base_model, force=force
    )
    payload = {
        "created": created,
        "profile": config.name,
        "path": str(paths.root),
        "home": str(paths.home),
        "base_model": config.base_model,
        "gguf_file": config.gguf_file,
        "models_dir": str(paths.models_dir),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    action = "Created" if created else "Already exists"
    console.print(f"{action}: [bold]{paths.root}[/bold]")
    console.print("Privacy: index, SFT JSONL, and adapters never leave this machine.")
    console.print(
        f"Default train model (quantized): {config.base_model} ({DEFAULT_MLX_SIZE_HINT})"
    )
    console.print(
        f"Default filter runtime: GGUF {config.gguf_file} ({DEFAULT_GGUF_SIZE_HINT}) "
        f"→ {paths.models_dir}"
    )
    console.print("Next: personality-protect download   # one ~5–7 GB quantized artifact")


@app.command("ingest")
def ingest_cmd(
    ctx: typer.Context,
    linkedin: Optional[Path] = typer.Option(
        None,
        "--linkedin",
        help="LinkedIn export directory or .zip (Shares*/Comments*/Articles*).",
    ),
    path: Optional[list[Path]] = typer.Option(
        None,
        "--path",
        help="Local email/doc/note file or directory (repeatable). Read in place.",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Source label for --path items: email | doc | note | demo.",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Register LinkedIn export and/or local writing paths into the index."""
    _banner_from_ctx(ctx, json_mode=as_json)
    if not linkedin and not path:
        console.print("[red]Provide --linkedin and/or --path.[/red]")
        raise typer.Exit(2)
    paths = get_paths(profile, home=home)
    try:
        load_config(paths)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        added, new_pieces = run_ingest(
            paths,
            linkedin=linkedin,
            local=list(path) if path else None,
            source_hint=source,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    all_pieces = load_index(paths.index_path)
    summary = summarize_by_source_year(all_pieces)
    payload = {
        "added": added,
        "batch": len(new_pieces),
        "index_total": len(all_pieces),
        "summary": summary,
        "index": str(paths.index_path),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"Added [bold]{added}[/bold] new pieces "
                  f"(batch parsed {len(new_pieces)}; index now {len(all_pieces)}).")
    console.print(f"Index: {paths.index_path}")
    _print_summary(summary, title="Index")


@app.command("select")
def select_cmd(
    ctx: typer.Context,
    min_words: int = typer.Option(
        DEFAULT_MIN_WORDS,
        "--min-words",
        help="Minimum word count (default >50 via 50).",
    ),
    through_year: int = typer.Option(
        DEFAULT_THROUGH_YEAR,
        "--through-year",
        help="Include pieces dated through this year (default 2024).",
    ),
    include_undated: bool = typer.Option(
        False,
        "--include-undated",
        help="Include pieces with no detectable date.",
    ),
    include: Optional[list[str]] = typer.Option(
        None,
        "--include",
        help="Force-include piece id (repeatable).",
    ),
    exclude: Optional[list[str]] = typer.Option(
        None,
        "--exclude",
        help="Exclude piece id (repeatable).",
    ),
    source: Optional[list[str]] = typer.Option(
        None,
        "--source",
        help="Only these sources (repeatable).",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Filter index and confirm counts by source/year before train."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    try:
        selection, selected = run_select(
            paths,
            min_words=min_words,
            through_year=through_year,
            include_undated=include_undated,
            include_ids=list(include) if include else None,
            exclude_ids=list(exclude) if exclude else None,
            sources=list(source) if source else None,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    payload = selection.to_dict()
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(
        f"Selected [bold]{len(selected)}[/bold] pieces "
        f"(min_words={selection.min_words}, through_year={selection.through_year})."
    )
    if selection.summary.get("undated_in_index"):
        console.print(
            f"Undated in index: {selection.summary['undated_in_index']} "
            f"(use --include-undated or --include <id>)."
        )
    _print_summary(selection.summary)
    console.print(f"Saved: {paths.selection_path}")
    console.print("Next: personality-protect train")


@app.command("download")
def download_cmd(
    ctx: typer.Context,
    format: str = typer.Option(
        "gguf",
        "--format",
        help="gguf (default, ~5.6 GB Q4_K_M) | mlx (~6 GB 4-bit)",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Download ONE quantized local model (~5–7 GB). Not full BF16."""
    _banner_from_ctx(ctx, json_mode=as_json)
    if format not in {"gguf", "mlx"}:
        console.print("[red]--format must be gguf or mlx[/red]")
        raise typer.Exit(2)
    if not as_json:
        hint = DEFAULT_GGUF_SIZE_HINT if format == "gguf" else DEFAULT_MLX_SIZE_HINT
        console.print(f"Downloading quantized {format} artifact ({hint})…")
    result = run_download(format=format, home=home, profile=profile)  # type: ignore[arg-type]
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        raise typer.Exit(0 if result.status in {"ok", "exists"} else 1)
    console.print(f"Status: [bold]{result.status}[/bold]")
    console.print(f"Path: {result.path}")
    console.print(f"Size: {result.size_hint}")
    if result.notes:
        console.print(result.notes)
    if result.status not in {"ok", "exists"}:
        raise typer.Exit(1)


@app.command("train")
def train_cmd(
    ctx: typer.Context,
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="auto | mlx | cuda | cpu | mock",
    ),
    max_steps: int = typer.Option(100, "--max-steps", help="Train steps/iters."),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Smoke-train without downloading any multi-GB model.",
    ),
    sft_only: bool = typer.Option(
        False,
        "--sft-only",
        help="Build local SFT JSONL only; skip weight training.",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Build SFT JSONL and run local LoRA on quantized base (or mock smoke train)."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    if backend not in {"auto", "mlx", "cuda", "cpu", "mock"}:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        raise typer.Exit(2)

    detected = detect_backend("mock" if mock else backend)  # type: ignore[arg-type]
    if not as_json and not sft_only:
        console.print(f"Backend: [bold]{detected}[/bold]")
        console.print(backend_docs(detected))

    try:
        result = run_train(
            paths,
            backend=backend,  # type: ignore[arg-type]
            max_steps=max_steps,
            mock=mock,
            sft_only=sft_only,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return
    console.print(f"Status: [bold]{result.status}[/bold] ({result.backend})")
    console.print(f"Examples: {result.examples}")
    console.print(f"Adapter dir: {result.adapter_dir}")
    if result.notes:
        console.print(result.notes)
    if result.status == "ok":
        console.print("Next: personality-protect filter --text '…'")


@app.command("filter")
def filter_cmd(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", help="Draft text to rewrite."),
    file: Optional[Path] = typer.Option(None, "--file", help="Read draft from file."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write rewrite to file."),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="auto | llama | gguf | mlx | transformers | mock",
    ),
    gguf: Optional[Path] = typer.Option(
        None,
        "--gguf",
        help=f"Path to local Q4/Q5 GGUF (default under models/{DEFAULT_GGUF_FILE}).",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Rewrite a draft with the local voice adapter (GGUF/llama.cpp preferred)."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    if backend not in {"auto", "llama", "gguf", "mlx", "transformers", "mock"}:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        raise typer.Exit(2)
    try:
        draft = read_draft_input(text, file)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    try:
        rewritten, used = filter_draft(
            draft, paths, backend=backend, gguf=gguf  # type: ignore[arg-type]
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if out:
        out.write_text(rewritten + "\n", encoding="utf-8")

    if as_json:
        typer.echo(json.dumps({"backend": used, "text": rewritten}, indent=2, ensure_ascii=False))
        return
    console.print(f"[dim]backend={used}[/dim]")
    console.print(rewritten)
    if out:
        console.print(f"[dim]wrote {out}[/dim]")


@app.command("demo")
def demo_cmd(
    ctx: typer.Context,
    home: Optional[Path] = typer.Option(
        None,
        "--home",
        help="State dir for demo profile (default under ~/.personality-protect).",
    ),
    draft: Optional[str] = typer.Option(
        None,
        "--draft",
        help="Optional frontier-style draft to filter.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run synthetic corpus end-to-end (safe for public screenshots)."""
    _banner_from_ctx(ctx, json_mode=as_json)
    result = run_demo(home=home, draft=draft)
    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    console.print("[bold]Demo complete[/bold] (synthetic data only; nothing left the machine).")
    console.print(f"Ingested: {result['ingested']}  Selected: {result['selected']}")
    console.print(f"Train: {result['train_status']} via {result['train_backend']}")
    console.print("")
    console.print("[bold]Draft[/bold]")
    console.print(result["draft"])
    console.print("")
    console.print("[bold]Filtered[/bold]")
    console.print(result["rewritten"])


@app.command("api")
def api_cmd(
    ctx: typer.Context,
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Bind address (localhost only)."),
    port: int = typer.Option(DEFAULT_PORT, "--port"),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
) -> None:
    """Stub local-only HTTP API for a future browser extension."""
    _banner_from_ctx(ctx)
    console.print("Local API stub — binds to loopback only; corpus never uploaded.")
    try:
        serve_api(host=host, port=port, profile=profile)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show local profile paths and counts."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home or default_home())
    info: dict = {
        "home": str(paths.home),
        "profile": profile,
        "root": str(paths.root),
        "exists": paths.config_path.is_file(),
        "index_pieces": 0,
        "has_selection": paths.selection_path.is_file(),
        "has_sft": paths.sft_jsonl.is_file(),
        "has_adapter": (paths.adapters_dir / "latest").is_dir(),
        "models_dir": str(paths.models_dir),
        "has_gguf": any(paths.models_dir.glob("*.gguf")) if paths.models_dir.is_dir() else False,
        "detected_backend": detect_backend("auto"),
    }
    if paths.config_path.is_file():
        cfg = load_config(paths)
        info["base_model"] = cfg.base_model
        info["gguf_file"] = cfg.gguf_file
        pieces = load_index(paths.index_path)
        info["index_pieces"] = len(pieces)
        info["summary"] = summarize_by_source_year(pieces)
    if as_json:
        typer.echo(json.dumps(info, indent=2))
        return
    for k, v in info.items():
        if k == "summary":
            continue
        console.print(f"{k}: {v}")
    if info.get("summary"):
        _print_summary(info["summary"], title="Index")


if __name__ == "__main__":
    app()
