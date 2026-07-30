"""PersonalityProtect CLI entrypoint."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from personality_protect import __version__
from personality_protect.api import DEFAULT_HOST, DEFAULT_PORT
from personality_protect.api import serve as serve_api
from personality_protect.config import (
    CORPUS_BLOCK_BELOW,
    CORPUS_WARN_BELOW,
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_SIZE_HINT,
    DEFAULT_MIN_WORDS,
    DEFAULT_MLX_SIZE_HINT,
    DEFAULT_PROFILE,
    DEFAULT_THROUGH_YEAR,
    DEFAULT_VOICE_MODE,
    DEFAULT_WRITE_ADAPTER,
    default_home,
    get_paths,
    init_profile,
    load_config,
)
from personality_protect.corpus_dedupe import DEFAULT_NEAR_RATIO, dedupe_pieces
from personality_protect.demo import run_demo
from personality_protect.download import run_download
from personality_protect.eval_compare import (
    list_synthetic_drafts,
    resolve_eval_draft,
    run_compare,
    run_eval,
    specificity_scorecard,
)
from personality_protect.eval_write_holdout import (
    run_eval_write_holdout,
    write_receipt,
)
from personality_protect.filter import (
    filter_draft,
    paragraph_windows,
    read_draft_input,
    rewrite_quality_flags,
    should_chunk_filter,
    suggest_max_tokens,
)
from personality_protect.ingest import run_ingest
from personality_protect.logo import (
    ColorMode,
    LogoDisplay,
    LogoMode,
    print_logo,
    should_show_logo,
)
from personality_protect.mlx_train import (
    DEFAULT_CHUNK_STEPS,
    DEFAULT_MAX_SEQ_LENGTH,
    PROOF_MAX_STEPS,
)
from personality_protect.models import load_index, save_index, summarize_by_source_year
from personality_protect.pair_gate import (
    MAX_INPUT_PROPER_PER_1K,
    MAX_STERILE_FRAG_DELTA,
    MAX_STERILE_PROPER_DELTA,
    MAX_STERILE_YOU_DELTA,
    MIN_FRAG_GAP_RATIO,
    MIN_MEDIAN_SENTENCE_GAP,
    gate_jsonl,
    gate_pair,
    sterile_flattener_check,
    write_kept_jsonl,
)
from personality_protect.select import run_select
from personality_protect.style_profile import run_build_style_profile
from personality_protect.train import (
    MockFallbackError,
    auto_max_steps,
    backend_docs,
    detect_backend,
    run_train,
)
from personality_protect.translator_eval import (
    load_packaged_author_holdout,
    load_packaged_foreign_holdout,
    score_translator_holdout,
)
from personality_protect.voice_index import build_voice_index
from personality_protect.write import (
    DEFAULT_WRITE_K,
    DEFAULT_WRITE_MAX_TOKENS,
    MAX_WRITE_K,
    MIN_WRITE_K,
    run_write,
)

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
            "Commands: init | download | ingest | dedupe-index | index-voice | "
            "build-style-profile | select | write | eval-write-holdout | train | filter | "
            "eval | compare | scorecard | pair-gate | sterile-check | "
            "translator-eval | demo | api | logo | status"
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


@app.command("index-voice")
def index_voice_cmd(
    ctx: typer.Context,
    holdout_id: Optional[list[str]] = typer.Option(
        None,
        "--holdout-id",
        help="Piece id to exclude from retrieval (repeatable).",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Rebuild the local voice retrieval index from the current corpus."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    try:
        load_config(paths)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    result = build_voice_index(paths, holdout_ids=holdout_id or ())
    if as_json:
        typer.echo(json.dumps(result, indent=2))
        return
    console.print(
        f"Indexed [bold]{result['indexed']}[/bold] voice exemplars "
        f"(skipped holdout: {result['skipped_holdout']})."
    )
    console.print(f"Voice index: {result['voice_index']}")


@app.command("dedupe-index")
def dedupe_index_cmd(
    ctx: typer.Context,
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Rewrite the index (default: report only).",
    ),
    near_ratio: float = typer.Option(
        DEFAULT_NEAR_RATIO,
        "--near-ratio",
        help="Similarity for near-identical captures; 1.0 for exact text only.",
    ),
    holdout_id: Optional[list[str]] = typer.Option(
        None,
        "--holdout-id",
        help="Id that must survive as keeper (repeatable; adds to the local holdout file).",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Drop pieces repeating another piece's text (re-ingest under a new path).

    Rebuild ``index-voice``, ``select`` and ``build-style-profile`` after
    applying, otherwise they keep counting the dropped pieces.
    """
    from personality_protect.writer_sft import load_holdout_id_set

    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    pieces = load_index(paths.index_path)
    if not pieces:
        console.print(f"[red]No corpus index at {paths.index_path}.[/red]")
        raise typer.Exit(1)

    holdouts = load_holdout_id_set(paths) | {
        str(piece_id) for piece_id in (holdout_id or []) if str(piece_id).strip()
    }
    result = dedupe_pieces(
        pieces,
        holdout_ids=holdouts,
        near_ratio=None if near_ratio >= 1.0 else near_ratio,
    )

    payload: dict[str, object] = {
        "index": str(paths.index_path),
        "applied": False,
        "near_ratio": near_ratio,
        "holdout_ids": sorted(holdouts),
        "before": summarize_by_source_year(pieces),
        "after": summarize_by_source_year(result.kept),
        **result.to_report(),
    }
    if apply and result.groups:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = paths.index_path.with_name(f"{paths.index_path.name}.bak-{stamp}")
        backup.write_bytes(paths.index_path.read_bytes())
        payload["backup"] = str(backup)
        payload["written"] = save_index(paths.index_path, result.kept)
        payload["applied"] = True

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(
        f"[bold]{len(result.groups)}[/bold] duplicate-text groups "
        f"({payload['exact_groups']} exact, {payload['near_groups']} near); "
        f"[bold]{payload['dropped']}[/bold] pieces removable."
    )
    console.print(f"by source: {payload['dropped_by_source']}")
    if result.groups and payload["cross_source_groups"]:
        console.print(
            f"[yellow]{len(payload['cross_source_groups'])} groups span sources "
            "— identical text kept under the more specific source.[/yellow]"
        )
    if payload["applied"]:
        console.print(
            f"Index now [bold]{payload['written']}[/bold] pieces. Backup: {payload['backup']}"
        )
        console.print("Rebuild: index-voice, select, build-style-profile.")
    elif result.groups:
        console.print("Report only. Pass --apply to rewrite the index.")


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
    force: bool = typer.Option(
        False,
        "--force",
        help=f"Allow continuing with fewer than {CORPUS_BLOCK_BELOW} pieces.",
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

    n = len(selected)
    gate_error = None
    gate_warn = None
    if n < CORPUS_BLOCK_BELOW and not force:
        gate_error = (
            f"Only {n} pieces selected (need >={CORPUS_BLOCK_BELOW} for full train). "
            "Ingest more writing or pass --force."
        )
    elif n < CORPUS_WARN_BELOW:
        gate_warn = (
            f"Only {n} pieces selected (recommend >={CORPUS_WARN_BELOW} "
            "for a credible voice adapter)."
        )

    payload = selection.to_dict()
    payload["corpus_warn"] = gate_warn
    payload["corpus_block"] = gate_error
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        if gate_error:
            raise typer.Exit(2)
        return
    console.print(
        f"Selected [bold]{n}[/bold] pieces "
        f"(min_words={selection.min_words}, through_year={selection.through_year})."
    )
    if selection.summary.get("undated_in_index"):
        console.print(
            f"Undated in index: {selection.summary['undated_in_index']} "
            f"(use --include-undated or --include <id>)."
        )
    _print_summary(selection.summary)
    if gate_warn:
        console.print(f"[yellow]{gate_warn}[/yellow]")
    if gate_error:
        console.print(f"[red]{gate_error}[/red]")
        raise typer.Exit(2)
    console.print(f"Saved: {paths.selection_path}")
    console.print("Next: personality-protect index-voice")
    console.print("Then: personality-protect build-style-profile")


@app.command("build-style-profile")
def build_style_profile_cmd(
    ctx: typer.Context,
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Compute corpus style stats + banned AI-filler list into style_profile.json."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    try:
        style, out_path = run_build_style_profile(paths)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    payload = {
        "path": str(out_path),
        "pieces": style["stats"]["pieces"],
        "words": style["stats"]["words"],
        "stats": style["stats"],
        "banned_ai_filler": style["banned_ai_filler"],
        "piece_ids": style["piece_ids"],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(
        f"Style profile from [bold]{style['stats']['pieces']}[/bold] pieces "
        f"({style['stats']['words']} words)."
    )
    stats = style["stats"]
    console.print(
        f"median_sentence={stats['median_sentence_words']} "
        f"short_line_ratio={stats['short_line_ratio']} "
        f"contraction_rate={stats['contraction_rate']} "
        f"you_gt_i={stats['you_gt_i']}"
    )
    console.print(f"banned_ai_filler: {len(style['banned_ai_filler'])} phrases")
    console.print(f"Saved: {out_path}")


@app.command("build-writer-sft")
def build_writer_sft_cmd(
    ctx: typer.Context,
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Build brief→post SFT JSONL for the writer LoRA (holdouts excluded)."""
    from personality_protect.writer_sft import run_build_writer_sft

    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    try:
        receipt = run_build_writer_sft(paths)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(receipt, indent=2, ensure_ascii=False))
        return
    console.print(
        f"writer SFT: {receipt['examples']} examples "
        f"(skipped {receipt['skipped']}) → {receipt['path']}"
    )


@app.command("write")
def write_cmd(
    ctx: typer.Context,
    topic: str = typer.Option(..., "--topic", help="What the post is about."),
    points: str = typer.Option(..., "--points", help="Facts/claims the post may use."),
    channel: str = typer.Option(
        "post",
        "--channel",
        help="post (default LinkedIn post) or article (outline→sections→stitch).",
    ),
    k: int = typer.Option(
        DEFAULT_WRITE_K,
        "--k",
        help=f"Exemplars to retrieve ({MIN_WRITE_K}–{MAX_WRITE_K}).",
    ),
    max_tokens: int = typer.Option(
        DEFAULT_WRITE_MAX_TOKENS,
        "--max-tokens",
        help="Generation budget for the draft (per section when --channel article).",
    ),
    no_adapter: bool = typer.Option(
        True,
        "--no-adapter/--adapter",
        help="Default: base weights. --adapter loads a local writer LoRA when present.",
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Write draft to file."),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Draft a post or article from retrieved exemplars (optional writer LoRA)."""
    _banner_from_ctx(ctx, json_mode=as_json)

    paths = get_paths(profile, home=home)
    try:
        load_config(paths)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        result = run_write(
            topic,
            points,
            paths,
            k=k,
            max_tokens=max_tokens,
            channel=channel,
            use_adapter=not no_adapter,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    guard_failed = bool(result["parrot_reject"] or result["invent_reject"])
    if as_json:
        # Prompt/messages carry the retrieved exemplars (personal text) and are
        # for local debugging only — never part of the emitted receipt.
        payload = {
            key: value
            for key, value in result.items()
            if key not in {"prompt", "messages", "exemplar_texts"}
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        if out and not guard_failed:
            out.write_text(result["text"] + "\n", encoding="utf-8")
        if guard_failed:
            raise typer.Exit(1)
        return

    console.print(
        f"[dim]voice_mode={result['voice_mode']} adapter={result['adapter']} "
        f"model={result['model']} k={result['k']} attempts={result['attempts']}[/dim]"
    )
    console.print(f"[dim]exemplars: {', '.join(result['exemplar_ids'])}[/dim]")
    if guard_failed:
        console.print(
            "[red]Guards still failing after one regen "
            f"(parrot={result['parrot_reject']} invent={result['invent_reject']} "
            f"entities+{len(result['invented_entities'])} "
            f"numbers+{len(result['invented_numbers'])}). "
            "Not saving draft — tighten --points or re-run.[/red]"
        )
        console.print(result["text"])
        raise typer.Exit(1)
    if out:
        out.write_text(result["text"] + "\n", encoding="utf-8")
        console.print(f"[dim]wrote {out}[/dim]")
    console.print(result["text"])


@app.command("eval-write-holdout")
def eval_write_holdout_cmd(
    ctx: typer.Context,
    holdout_id: Optional[list[str]] = typer.Option(
        None,
        "--holdout-id",
        help="Holdout piece id never indexed (repeatable).",
    ),
    k: int = typer.Option(
        DEFAULT_WRITE_K,
        "--k",
        help=f"Exemplars to retrieve for RAG drafts ({MIN_WRITE_K}–{MAX_WRITE_K}).",
    ),
    max_tokens: int = typer.Option(
        DEFAULT_WRITE_MAX_TOKENS,
        "--max-tokens",
        help="Generation budget per draft.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Write Contoso-safe receipt JSON (no draft bodies).",
    ),
    save_raw: bool = typer.Option(
        False,
        "--save-raw/--no-save-raw",
        help="Dump exact prompts and raw drafts under the profile's "
        "gitignored dogfood/raw dir (personal text; never commit).",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Score RAG write vs bare-base on never-indexed holdouts.

    Receipts omit draft/holdout bodies (ids + scores + invent counts only).
    Generation uses injectable MLX behind PP_MLX_ALLOW; tests mock generate.
    """
    _banner_from_ctx(ctx, json_mode=as_json)
    ids = [str(piece_id) for piece_id in (holdout_id or []) if str(piece_id).strip()]
    if not ids:
        console.print("[red]Provide at least one --holdout-id.[/red]")
        raise typer.Exit(2)

    paths = get_paths(profile, home=home)
    try:
        load_config(paths)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        receipt = run_eval_write_holdout(
            paths,
            ids,
            k=k,
            max_tokens=max_tokens,
            save_raw=save_raw,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if out:
        write_receipt(receipt, out)

    if as_json:
        typer.echo(json.dumps(receipt, indent=2, ensure_ascii=False))
        return

    wins = receipt["wins"]
    console.print(
        f"[bold]eval-write-holdout[/bold] n={receipt['n_holdouts']} "
        f"adapter={receipt['adapter']} model={receipt['model']}"
    )
    console.print(
        f"carve_ok={receipt['carve']['ok']} "
        f"wins rag={wins['rag']} base={wins['base']} tie={wins['tie']} "
        f"rag_beats_base={receipt['rag_beats_base']}"
    )
    for item in receipt["items"]:
        console.print(
            f"  {item['holdout_id']}: winner={item['winner']} "
            f"Δ={item['delta_base_minus_rag']} "
            f"rag_dist={item['rag_distance']} base_dist={item['base_distance']} "
            f"invent rag={item['rag_invent_reject']} base={item['base_invent_reject']} "
            f"brief_leak={item['brief_leakage_ratio']}"
        )
    if out:
        console.print(f"[dim]wrote receipt {out}[/dim]")
    if save_raw:
        console.print(
            "[dim]raw prompts/drafts under profile dogfood/raw (personal text — never commit)[/dim]"
        )


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
    max_steps: Optional[int] = typer.Option(
        None,
        "--max-steps",
        help="Train steps/iters (default: auto from SFT count, or smoke low-step).",
    ),
    chunk_steps: int = typer.Option(
        DEFAULT_CHUNK_STEPS,
        "--chunk-steps",
        help="MLX: iters per subprocess chunk (releases Metal memory between chunks).",
    ),
    memory_gb: Optional[float] = typer.Option(
        None,
        "--memory-gb",
        help="MLX: cap Metal wired memory in GB (default: ~40% of RAM, max 20 GB).",
    ),
    max_seq_length: int = typer.Option(
        DEFAULT_MAX_SEQ_LENGTH,
        "--max-seq-length",
        min=128,
        help=(
            "MLX: prompt+input+target token window. Default 1024 for post pairs; "
            "use 2048 with a higher --memory-gb cap for article sections."
        ),
    ),
    proof: bool = typer.Option(
        False,
        "--proof",
        help=f"Bounded real train ({PROOF_MAX_STEPS} steps) for receipts — not mock.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "MLX: continue from adapters.safetensors + train_chunks.json "
            "(skips completed_steps; does not delete weights). "
            "Incomplete checkpoints auto-resume even without this flag."
        ),
    ),
    force_retrain: bool = typer.Option(
        False,
        "--force-retrain",
        help="MLX: delete existing adapters/checkpoints and start a clean train.",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="CI/low-step train (does not silently substitute mock).",
    ),
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Permit mock fallback when a real backend is unavailable.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Smoke-train without downloading any multi-GB model.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=f"Allow train with fewer than {CORPUS_BLOCK_BELOW} SFT examples.",
    ),
    sft_only: bool = typer.Option(
        False,
        "--sft-only",
        help="Build local SFT JSONL only; skip weight training.",
    ),
    pairs: Optional[Path] = typer.Option(
        None,
        "--pairs",
        help=(
            "Gated flatten→author JSONL (pairs.kept.jsonl). "
            "Voice-pair mode: translator SFT only; skips leave_alone/identity minting."
        ),
    ),
    writer: bool = typer.Option(
        False,
        "--writer",
        help="Build brief→post writer SFT and train a writer LoRA (RAG write path).",
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

    if pairs is not None and writer:
        console.print("[red]Pass only one of --writer or --pairs.[/red]")
        raise typer.Exit(2)
    if pairs is not None and not pairs.is_file():
        console.print(f"[red]pairs file not found: {pairs}[/red]")
        raise typer.Exit(2)

    try:
        detected = detect_backend(
            "mock" if mock else backend,  # type: ignore[arg-type]
            allow_mock=allow_mock or mock,
        )
    except MockFallbackError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not as_json and not sft_only:
        console.print(f"Backend: [bold]{detected}[/bold]")
        console.print(backend_docs(detected))
        if pairs is not None:
            console.print(f"Voice-pair mode: translator SFT from {pairs}")
        if proof and max_steps is None and not smoke and not mock:
            console.print(
                f"Steps: proof mode ({PROOF_MAX_STEPS}) in chunks of {chunk_steps}"
            )
        elif max_steps is None and not smoke and not mock:
            console.print(
                f"Steps: auto (≈{auto_max_steps(100)} for 100 examples; "
                f"scaled from your SFT count), chunks of {chunk_steps}"
            )
        elif smoke or mock:
            console.print(
                f"Steps: smoke/low-step ({max_steps or auto_max_steps(1, smoke=True)})"
            )
        if memory_gb is not None:
            console.print(f"Metal wired memory cap: {memory_gb} GB")
        console.print(f"MLX max sequence length: {max_seq_length}")
        if resume and force_retrain:
            console.print("[red]Pass only one of --resume or --force-retrain[/red]")
            raise typer.Exit(2)
        if force_retrain:
            console.print("Force retrain: wiping adapters and starting clean")
        else:
            from personality_protect.mlx_train import (
                completed_steps_from_meta,
                is_incomplete_checkpoint,
                load_train_checkpoint_meta,
            )

            latest = paths.adapters_dir / "latest"
            prior = load_train_checkpoint_meta(latest) or {}
            done = completed_steps_from_meta(prior)
            incomplete = is_incomplete_checkpoint(latest)
            if resume or incomplete:
                target = max_steps or int(prior.get("total_steps") or 0) or "…"
                why = "explicit --resume" if resume else "incomplete checkpoint (auto-resume)"
                console.print(
                    f"Resume: continuing from step {done}/{target} "
                    f"({why}; adapters.safetensors + train_chunks.json). "
                    "Each finished chunk is a checkpoint — crash-safe with --resume."
                )
            elif (latest / "adapters.safetensors").is_file() and not resume:
                console.print(
                    "Note: existing adapters will be wiped unless you pass "
                    "--resume (or an incomplete train_chunks.json auto-resumes). "
                    "Use --force-retrain to wipe deliberately."
                )
    progress_holder: dict = {}

    def _make_plain_progress_callback():
        """Line-oriented progress for nohup / redirected logs (no Rich Live)."""

        def on_progress(info: dict) -> None:
            kind = info.get("kind")
            if kind == "start":
                done = info.get("completed_steps") or 0
                total = info.get("total_steps") or 0
                print(
                    f"MLX LoRA start: {total} steps · "
                    f"{info.get('chunks')} chunks · "
                    f"cap {info.get('wired_limit_gb')} GB · "
                    f"seq {info.get('max_seq_length')} · "
                    f"resume={info.get('resume')} · "
                    f"from step {done}/{total}",
                    flush=True,
                )
            elif kind == "chunk_start":
                print(
                    f"chunk {info.get('chunk')}/{info.get('chunks')} "
                    f"start ({info.get('chunk_iters')} iters, "
                    f"completed {info.get('completed_steps')}/"
                    f"{info.get('total_steps')})",
                    flush=True,
                )
            elif kind == "step":
                line = str(info.get("line") or "").strip()
                if line:
                    print(
                        f"step {info.get('global_step')}/"
                        f"{info.get('total_steps')}: {line}",
                        flush=True,
                    )
            elif kind == "chunk_done":
                peak = info.get("peak_mem_gb")
                peak_s = f" peak_mem={peak:.1f}GB" if peak else ""
                print(
                    f"chunk {info.get('chunk')}/{info.get('chunks')} done "
                    f"({info.get('completed_steps')}/"
                    f"{info.get('total_steps')}){peak_s}",
                    flush=True,
                )
            elif kind == "done":
                print(
                    f"MLX LoRA done: adapter saved "
                    f"(wired_cap={info.get('wired_limit_gb')} GB, "
                    f"peak={info.get('peak_mem_gb')})",
                    flush=True,
                )
            elif kind == "error":
                detail = str(info.get("detail") or "")[-500:]
                print(
                    f"MLX LoRA FAILED chunk={info.get('chunk')} "
                    f"exit={info.get('returncode')}\n{detail}",
                    flush=True,
                )

        return on_progress

    def _make_progress_callback():
        if sft_only or detected != "mlx":
            return None

        # Rich Live + subprocess PIPE is hostile to nohup/redirected logs:
        # the monitor shows an empty file after the banner even while (or after)
        # the worker dies. Prefer plain flushed lines when stdout is not a TTY
        # or when --json is used (final JSON still prints after training).
        if as_json or not sys.stdout.isatty():
            return _make_plain_progress_callback()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[detail]}"),
            console=console,
            transient=False,
        )
        progress.start()
        task_id = progress.add_task("train", total=None, detail="")
        progress_holder["progress"] = progress
        progress_holder["task_id"] = task_id

        def on_progress(info: dict) -> None:
            kind = info.get("kind")
            tid = progress_holder["task_id"]
            if kind == "start":
                progress.update(
                    tid,
                    total=info.get("total_steps") or 0,
                    completed=0,
                    description="MLX LoRA",
                    detail=(
                        f"cap {info.get('wired_limit_gb')} GB · "
                        f"{info.get('chunks')} chunks"
                    ),
                )
            elif kind == "chunk_start":
                progress.update(
                    tid,
                    detail=(
                        f"chunk {info.get('chunk')}/{info.get('chunks')} "
                        f"({info.get('chunk_iters')} iters)"
                    ),
                )
            elif kind == "step":
                progress.update(
                    tid,
                    completed=info.get("global_step") or 0,
                    detail=str(info.get("line") or "")[:60],
                )
            elif kind == "chunk_done":
                peak = info.get("peak_mem_gb")
                peak_s = f" · peak {peak:.1f} GB" if peak else ""
                progress.update(
                    tid,
                    completed=info.get("completed_steps") or 0,
                    detail=f"chunk {info.get('chunk')} done{peak_s}",
                )
            elif kind == "done":
                progress.update(
                    tid,
                    completed=info.get("total_steps") or 0,
                    detail="adapter saved",
                )
            elif kind == "error":
                progress.update(tid, detail="FAILED")

        return on_progress

    # Stream MLX progress even with --json so redirected/nohup logs aren't empty.
    show_mlx_progress = (
        not sft_only
        and not mock
        and backend != "mock"
        and detected == "mlx"
    )
    callback = _make_progress_callback() if show_mlx_progress else None

    try:
        result = run_train(
            paths,
            backend=backend,  # type: ignore[arg-type]
            max_steps=max_steps,
            mock=mock,
            smoke=smoke,
            allow_mock=allow_mock,
            force=force,
            sft_only=sft_only,
            chunk_steps=chunk_steps,
            memory_gb=memory_gb,
            max_seq_length=max_seq_length,
            proof=proof,
            resume=resume,
            force_retrain=force_retrain,
            progress_callback=callback,
            pairs=pairs,
            writer=writer,
        )
    except (FileNotFoundError, RuntimeError, MockFallbackError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        prog = progress_holder.get("progress")
        if prog is not None:
            prog.stop()

    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return
    console.print(f"Status: [bold]{result.status}[/bold] ({result.backend})")
    console.print(f"Examples: {result.examples}  Steps: {result.steps}")
    console.print(f"Adapter dir: {result.adapter_dir}")
    if result.notes:
        console.print(result.notes)
    if result.status == "ok":
        console.print("Next: personality-protect eval-write-holdout --out receipt.json")
        console.print("[dim]Keep this adapter only if it beats RAG-alone on the holdout.[/dim]")


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
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Generation budget (default: scales with draft length, up to 4096).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Fail if the translator returns a byte-identical echo.",
    ),
    chunk: Optional[bool] = typer.Option(
        None,
        "--chunk/--no-chunk",
        help="Chunk article-length drafts (default: auto when draft > 1600 chars).",
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

    budget = suggest_max_tokens(draft, override=max_tokens)
    auto_chunk = should_chunk_filter(draft) if chunk is None else bool(chunk)
    n_windows = len(paragraph_windows(draft)) if auto_chunk else 1
    try:
        rewritten, used = filter_draft(
            draft,
            paths,
            backend=backend,  # type: ignore[arg-type]
            max_tokens=budget,
            gguf=gguf,
            force=force,
            chunk=chunk,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    flags = rewrite_quality_flags(draft, rewritten)
    chunked = bool(auto_chunk and n_windows > 1)
    # --force means "must rewrite". Byte-identical output is a hard failure —
    # do not write a fake *-voiced.md that is just the Claude draft.
    force_echo = bool(force and flags.get("unchanged"))
    # Translator may invent diction, never entities/figures absent from source.
    invent_reject = bool(flags.get("invented_facts"))

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "backend": used,
                    "text": rewritten,
                    "max_tokens": budget,
                    "force": force,
                    "chunked": chunked,
                    "chunk_windows": n_windows if auto_chunk else 1,
                    "force_echo_reject": force_echo,
                    "invent_reject": invent_reject,
                    **flags,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if force_echo or invent_reject:
            raise typer.Exit(1)
        if out:
            out.write_text(rewritten + "\n", encoding="utf-8")
        return

    chunk_note = f" chunked={n_windows}windows" if chunked else ""
    console.print(
        f"[dim]backend={used} max_tokens={budget} force={force}{chunk_note}[/dim]"
    )
    if force_echo:
        console.print(
            "[red]--force produced byte-identical output (echo). "
            "Not writing voiced file — filter did not rewrite.[/red]"
        )
        raise typer.Exit(1)
    if invent_reject:
        console.print(
            "[red]Rewrite invented proper nouns or evidence figures "
            f"(proper+{flags.get('invented_proper_count', 0)} "
            f"numbers+{flags.get('invented_number_count', 0)}). "
            "Not writing voiced file — keep source entities/figures only.[/red]"
        )
        raise typer.Exit(1)
    if flags["unchanged"]:
        console.print(
            "[yellow]Filter left draft unchanged "
            "(model echo or truncation guard).[/yellow]"
        )
    if flags["likely_truncated"]:
        console.print(
            "[red]Rewrite looks truncated — raise --max-tokens and re-run.[/red]"
        )
    if out:
        out.write_text(rewritten + "\n", encoding="utf-8")
        console.print(f"[dim]wrote {out}[/dim]")
    console.print(rewritten)


@app.command("scorecard")
def scorecard_cmd(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", help="Draft text to score."),
    file: Optional[Path] = typer.Option(None, "--file", help="Read draft from file."),
    synthetic: Optional[str] = typer.Option(
        None,
        "--synthetic",
        help="Packaged synthetic draft stem (see data/evals/).",
    ),
    channel: Optional[str] = typer.Option(
        None,
        "--channel",
        help="linkedin|article — floors: proper 20 vs 42; numbers "
        "advisory(0) on both. Default: infer from word count (<500 → linkedin).",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Gate a draft on specificity before voicing (no model call).

    Hard FAIL: proper nouns /1k (channel p10). Numbers /1k are advisory only.
    Advisory: numbers, median sentence, short-line ratio, you vs I.
    """
    _banner_from_ctx(ctx, json_mode=as_json)
    try:
        draft, label = resolve_eval_draft(text=text, file=file, synthetic=synthetic)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    card = specificity_scorecard(draft, channel=channel)
    card["label"] = label
    if as_json:
        typer.echo(json.dumps(card, indent=2, ensure_ascii=False))
        raise typer.Exit(0 if card["pass"] else 1)

    console.print(f"[bold]Scorecard[/bold] label={label} words={card['words']}")
    num_need = card["thresholds"]["min_numbers_per_1k"]
    num_note = (
        f"(need ≥{num_need})"
        if float(num_need) > 0
        else "(advisory — no hard gate)"
    )
    console.print(
        f"proper_nouns/1k={card['proper_nouns_per_1k']} "
        f"(need ≥{card['thresholds']['min_proper_per_1k']}) "
        f"numbers/1k={card['numbers_per_1k']} {num_note}"
    )
    console.print(
        f"[dim]advisory median_sentence={card['median_sentence_words']} "
        f"short_line_ratio={card['short_line_ratio']} "
        f"you={card['you_count']} i={card['i_count']}[/dim]"
    )
    if card["pass"]:
        console.print("[green]PASS[/green] — draft clears specificity gates.")
    else:
        console.print(
            f"[red]FAIL[/red] — {', '.join(card['failed'])}. "
            "Do not voice yet; fix the drafter/brief."
        )
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", help="Draft text to eval."),
    file: Optional[Path] = typer.Option(None, "--file", help="Read draft from file."),
    synthetic: Optional[str] = typer.Option(
        None,
        "--synthetic",
        help="Packaged synthetic draft stem (see data/evals/).",
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="auto | llama | gguf | mlx | transformers | mock",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Write before/after filter receipts under the local profile evals/ dir."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    try:
        draft, label = resolve_eval_draft(text=text, file=file, synthetic=synthetic)
    except (ValueError, FileNotFoundError) as exc:
        if text is None and file is None and synthetic is None:
            names = ", ".join(p.stem for p in list_synthetic_drafts()) or "(none)"
            console.print(f"[red]{exc}[/red]")
            console.print(f"Packaged synthetics: {names}")
            raise typer.Exit(2) from exc
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    try:
        result = run_eval(paths, draft, backend=backend, label=label)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]Eval[/bold] backend={result['backend']}")
    console.print(f"slop before={result['slop_before']} after={result['slop_after']}")
    console.print(f"Wrote: {result['dir']}")
    console.print("")
    console.print("[bold]After[/bold]")
    console.print(result["rewritten"])


@app.command("compare")
def compare_cmd(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", help="Draft text to compare."),
    file: Optional[Path] = typer.Option(None, "--file", help="Read draft from file."),
    synthetic: Optional[str] = typer.Option(
        None,
        "--synthetic",
        help="Packaged synthetic draft stem (default: first under data/evals/).",
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Filter backend for LoRA path: auto | llama | mlx | mock | …",
    ),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile"),
    home: Optional[Path] = typer.Option(None, "--home"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Compare raw draft vs prompt few-shot baseline vs LoRA/mock filter."""
    _banner_from_ctx(ctx, json_mode=as_json)
    paths = get_paths(profile, home=home)
    try:
        draft, label = resolve_eval_draft(text=text, file=file, synthetic=synthetic)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    try:
        result = run_compare(paths, draft, backend=backend, label=label)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]Compare[/bold] filter_backend={result['filter_backend']}")
    console.print(
        f"slop raw={result['slop']['raw']} "
        f"prompt={result['slop']['prompt_baseline']} "
        f"lora={result['slop']['lora']}"
    )
    console.print(f"Wrote: {result['dir']}")
    console.print("")
    console.print("[bold]LoRA / adapter path[/bold]")
    console.print(result["lora"])


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
    """Optional synthetic smoke tour of the write path (no model download)."""
    _banner_from_ctx(ctx, json_mode=as_json)
    result = run_demo(home=home, draft=draft)
    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    console.print(
        "[bold]Smoke tour complete[/bold] "
        "(synthetic Contoso corpus; the model call inside write is stubbed)."
    )
    console.print(
        "[dim]Product path: ingest → select → index-voice → "
        "build-style-profile → write.[/dim]"
    )
    console.print(
        "[dim]For a real draft: download --format mlx, then "
        "write --topic … --points … on Apple Silicon.[/dim]"
    )
    console.print(
        f"Ingested: {result['ingested']}  Selected: {result['selected']}  "
        f"Indexed: {result['indexed']}"
    )
    console.print(
        f"Style card: {result['style_pieces']} pieces, "
        f"{result['banned_ai_filler']} banned filler phrases"
    )
    console.print("")
    exemplars = result.get("write_exemplars") or []
    console.print(
        f"write channel={result['write_channel']} "
        f"adapter={result['write_adapter']} model=stub "
        f"exemplars={len(exemplars)}"
    )
    console.print(result["written"])
    console.print("")
    console.print("[dim]Legacy mock train/filter (not the drafting path):[/dim]")
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


@app.command("pair-gate")
def pair_gate_cmd(
    ctx: typer.Context,
    pairs: Optional[Path] = typer.Option(
        None,
        "--pairs",
        help="JSONL of voice pairs (input/output, not_you/author, …).",
    ),
    input_text: Optional[str] = typer.Option(
        None, "--input-text", help="Single-pair input (not-you) text."
    ),
    output_text: Optional[str] = typer.Option(
        None, "--output-text", help="Single-pair output (author voice) text."
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input-file", help="Read single-pair input from file."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output-file", help="Read single-pair output from file."
    ),
    keep_out: Optional[Path] = typer.Option(
        None,
        "--keep-out",
        help="Write kept JSONL rows here (batch mode only).",
    ),
    drop_log: Optional[Path] = typer.Option(
        None,
        "--drop-log",
        help="Write dropped rows + reasons as JSON here (batch mode).",
    ),
    channel: str = typer.Option(
        "auto",
        "--channel",
        help="Gate channel: post, article, or auto (metadata first).",
    ),
    max_input_proper_1k: float = typer.Option(
        MAX_INPUT_PROPER_PER_1K,
        "--max-input-proper-1k",
        help="Fail if flattened input proper/1k exceeds this.",
    ),
    min_frag_gap_ratio: float = typer.Option(
        MIN_FRAG_GAP_RATIO,
        "--min-frag-gap-ratio",
        help="Fail if output short-line ratio − input is below this.",
    ),
    min_median_sentence_gap: float = typer.Option(
        MIN_MEDIAN_SENTENCE_GAP,
        "--min-median-sentence-gap",
        help="Fail if input median sentence length − output is below this.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Drop poisoned flatten→voice pairs before training.

    Both channels enforce input proper/1k, input you>i false, and input median
    sentence meaningfully longer than output. Post also enforces fragment-gap;
    article explicitly skips that post-specific rhythm check.
    """
    _banner_from_ctx(ctx, json_mode=as_json)
    channel = channel.strip().lower()
    if channel not in {"post", "article", "auto"}:
        console.print("[red]--channel must be post, article, or auto[/red]")
        raise typer.Exit(2)
    kwargs = {
        "channel": channel,
        "max_input_proper_1k": max_input_proper_1k,
        "min_frag_gap_ratio": min_frag_gap_ratio,
        "min_median_sentence_gap": min_median_sentence_gap,
    }

    if pairs is not None:
        if not pairs.is_file():
            console.print(f"[red]pairs file not found: {pairs}[/red]")
            raise typer.Exit(2)
        try:
            report = gate_jsonl(pairs, **kwargs)
        except (ValueError, json.JSONDecodeError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        if keep_out is not None:
            write_kept_jsonl(report["kept_rows"], keep_out)
        if drop_log is not None:
            drop_log.write_text(
                json.dumps(
                    {
                        "dropped": report["dropped"],
                        "rows": [
                            {
                                "line": r["line"],
                                "resolved_channel": r["resolved_channel"],
                                "applied_checks": r["applied_checks"],
                                "skipped_checks": r["skipped_checks"],
                                "thresholds": r["thresholds"],
                                "failed": r["failed"],
                                "reasons": r["reasons"],
                                "axes": r["axes"],
                            }
                            for r in report["dropped_rows"]
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        summary = {
            "path": report["path"],
            "total": report["total"],
            "kept": report["kept"],
            "dropped": report["dropped"],
            "requested_channel": report["requested_channel"],
            "resolved_channels": report["resolved_channels"],
            "thresholds": report["thresholds"],
            "drop_reasons": {},
        }
        for row in report["dropped_rows"]:
            for code in row["failed"]:
                summary["drop_reasons"][code] = (
                    summary["drop_reasons"].get(code, 0) + 1
                )
        if as_json:
            typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            console.print(
                f"[bold]Pair gate[/bold] total={summary['total']} "
                f"kept={summary['kept']} dropped={summary['dropped']}"
            )
            if summary["drop_reasons"]:
                console.print(f"drop_reasons={summary['drop_reasons']}")
            if keep_out is not None:
                console.print(f"kept → {keep_out}")
            if drop_log is not None:
                console.print(f"drop log → {drop_log}")
        raise typer.Exit(0 if report["dropped"] == 0 else 1)

    inp = input_text
    out = output_text
    if input_file is not None:
        inp = input_file.read_text(encoding="utf-8")
    if output_file is not None:
        out = output_file.read_text(encoding="utf-8")
    if inp is None or out is None:
        console.print(
            "[red]Provide --pairs JSONL, or both input and output "
            "(--input-text/--output-text or --input-file/--output-file).[/red]"
        )
        raise typer.Exit(2)

    result = gate_pair(inp, out, **kwargs)
    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        console.print(
            f"[bold]Pair gate[/bold] "
            f"{'PASS' if result['pass'] else 'FAIL'} "
            f"channel={result['resolved_channel']}"
        )
        console.print(
            f"input proper/1k={result['input']['proper_per_1k']} "
            f"you_gt_i={result['input']['you_gt_i']} "
            f"median={result['input']['median_sentence_words']}"
        )
        console.print(
            f"output short_line={result['output']['short_line_ratio']} "
            f"median={result['output']['median_sentence_words']} "
            f"frag_gap={result['frag_gap_ratio']} "
            f"median_gap={result['median_sentence_gap']}"
        )
        if not result["pass"]:
            console.print(f"[red]failed={result['failed']}[/red]")
            for code, msg in result["reasons"].items():
                console.print(f"  {code}: {msg}")
    raise typer.Exit(0 if result["pass"] else 1)


@app.command("sterile-check")
def sterile_check_cmd(
    ctx: typer.Context,
    author_flat: Path = typer.Option(
        ...,
        "--author-flat",
        help="Flatten of one author post (same prompt as foreign).",
    ),
    foreign_flat: Path = typer.Option(
        ...,
        "--foreign-flat",
        help="Flatten of a press release you did not write.",
    ),
    max_proper_delta: float = typer.Option(
        MAX_STERILE_PROPER_DELTA, "--max-proper-delta"
    ),
    max_frag_delta: float = typer.Option(
        MAX_STERILE_FRAG_DELTA, "--max-frag-delta"
    ),
    max_you_delta: int = typer.Option(MAX_STERILE_YOU_DELTA, "--max-you-delta"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Contamination preflight before bulk pair generation.

    Flatten one author post and one foreign press release with the same
    sterile prompt. If the author-derived flatten scores higher on proper
    nouns, fragments, or you/I, the flattener is leaking — do not generate
    hundreds of pairs until this passes.
    """
    _banner_from_ctx(ctx, json_mode=as_json)
    if not author_flat.is_file() or not foreign_flat.is_file():
        console.print("[red]author-flat and foreign-flat must be existing files[/red]")
        raise typer.Exit(2)
    result = sterile_flattener_check(
        author_flat.read_text(encoding="utf-8"),
        foreign_flat.read_text(encoding="utf-8"),
        max_proper_delta=max_proper_delta,
        max_frag_delta=max_frag_delta,
        max_you_delta=max_you_delta,
    )
    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        console.print(
            f"[bold]Sterile flattener check[/bold] "
            f"{'PASS' if result['pass'] else 'FAIL'}"
        )
        console.print(f"deltas={result['deltas']}")
        if result["pass"]:
            console.print(
                "[green]PASS[/green] — flattener looks sterile; safe to bulk-pair."
            )
        else:
            console.print(
                f"[red]FAIL[/red] — {', '.join(result['failed'])}. "
                "Do not generate the training set; fix the flatten prompt/model."
            )
            for code, msg in result["reasons"].items():
                console.print(f"  {code}: {msg}")
    raise typer.Exit(0 if result["pass"] else 1)


@app.command("translator-eval")
def translator_eval_cmd(
    ctx: typer.Context,
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        help="Translator rewrite to score (required).",
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file",
        help="Sterile / foreign holdout input (omit with --packaged).",
    ),
    author_band: Optional[Path] = typer.Option(
        None,
        "--author-band",
        help="Held-out author post for voice-band axes (omit with --packaged).",
    ),
    packaged: bool = typer.Option(
        False,
        "--packaged",
        help="Use Contoso-safe packaged foreign + author-holdout fixtures.",
    ),
    min_frag_gap_ratio: float = typer.Option(
        MIN_FRAG_GAP_RATIO,
        "--min-frag-gap-ratio",
        help="Fail if output short-line ratio − input is below this.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Score a voice-translator rewrite against holdout axes.

    Success = proper / fragments / you·I move toward the author band vs the
    sterile input. Failure = byte-identical echo or still press-release band.
    Ear-test remains operator judgment; this automates axis checks only.
    Contoso-safe packaged fixtures — no personal data.
    """
    _banner_from_ctx(ctx, json_mode=as_json)
    if not output_file.is_file():
        console.print(f"[red]output file not found: {output_file}[/red]")
        raise typer.Exit(2)

    if packaged:
        sterile = load_packaged_foreign_holdout()
        author = load_packaged_author_holdout()
    else:
        if input_file is None or author_band is None:
            console.print(
                "[red]Provide --input-file and --author-band, or use --packaged.[/red]"
            )
            raise typer.Exit(2)
        if not input_file.is_file() or not author_band.is_file():
            console.print("[red]input-file and author-band must be existing files[/red]")
            raise typer.Exit(2)
        sterile = input_file.read_text(encoding="utf-8")
        author = author_band.read_text(encoding="utf-8")

    result = score_translator_holdout(
        sterile,
        output_file.read_text(encoding="utf-8"),
        author,
        min_frag_gap_ratio=min_frag_gap_ratio,
    )
    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        console.print(
            f"[bold]Translator holdout eval[/bold] "
            f"{'PASS' if result['pass'] else 'FAIL'}"
        )
        console.print(f"axes_moved={result['axes_moved']}")
        console.print(
            f"frag_gap={result['frag_gap_ratio']} "
            f"echo={result['echo']}"
        )
        if result["pass"]:
            console.print(
                "[green]PASS[/green] — axes moved toward author band "
                "(ear-test still operator judgment)."
            )
        else:
            console.print(
                f"[red]FAIL[/red] — {', '.join(result['failed'])}."
            )
            for code, msg in result["reasons"].items():
                console.print(f"  {code}: {msg}")
    raise typer.Exit(0 if result["pass"] else 1)


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
        "voice_mode": DEFAULT_VOICE_MODE,
        "adapter": "none",
        "write_adapter": DEFAULT_WRITE_ADAPTER,
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
        info["voice_mode"] = cfg.voice_mode
        info["write_adapter"] = cfg.write_adapter
        # Camp A RAG write path always runs base weights (adapter=none).
        info["adapter"] = "none"
        pieces = load_index(paths.index_path)
        info["index_pieces"] = len(pieces)
        info["summary"] = summarize_by_source_year(pieces)
    if as_json:
        typer.echo(json.dumps(info, indent=2))
        return
    for k, v in info.items():
        if k == "summary":
            continue
        display = "none" if v is None and k == "write_adapter" else v
        console.print(f"{k}: {display}")
    if info.get("summary"):
        _print_summary(info["summary"], title="Index")


if __name__ == "__main__":
    app()
