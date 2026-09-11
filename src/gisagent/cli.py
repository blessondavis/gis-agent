"""Command line entry point.

Everything the web UI does is available headlessly here, which is what makes the
container useful for batch work and CI as well as for interactive demos.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from gisagent.config import get_settings

app = typer.Typer(
    add_completion=False,
    help="Agentic road annotation on satellite imagery.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def doctor() -> None:
    """Check that the model, GPU, QGIS and LLM endpoint are all reachable."""
    settings = get_settings()
    table = Table("component", "status", "detail", title="gis-agent environment")

    def row(name: str, ok: bool | None, detail: str) -> None:
        mark = {True: "[green]ok[/]", False: "[red]fail[/]", None: "[yellow]warn[/]"}[ok]
        table.add_row(name, mark, detail)

    # --- torch / GPU ---
    try:
        import torch

        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            cap = "sm_%d%d" % torch.cuda.get_device_capability(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            row("torch", True,
                f"{torch.__version__} cuda={torch.version.cuda} "
                f"{torch.cuda.get_device_name(i)} {cap} {vram:.1f} GB")
            if cap not in torch.cuda.get_arch_list():
                row("gpu kernels", False,
                    f"{cap} missing from build ({', '.join(torch.cuda.get_arch_list())})")
        else:
            row("torch", None, f"{torch.__version__} (CPU only - inference will be slow)")
    except Exception as exc:
        row("torch", False, str(exc)[:80])

    # --- QGIS ---
    try:
        from gisagent.qgis.process import QgisProcess, available

        if available():
            row("qgis", True, QgisProcess().version().splitlines()[0][:70])
        else:
            row("qgis", False, "qgis_process not found on PATH")
    except Exception as exc:
        row("qgis", False, str(exc)[:80])

    # --- SAM 3 weights ---
    row("sam model", settings.sam_ready() or None,
        f"{settings.sam_model} ({'token set' if settings.hf_token else 'no HF_TOKEN'})")

    # --- LLM ---
    if not settings.llm_ready():
        row("llm", False, "OPENAI_API_KEY not set")
    else:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key,
                            base_url=settings.openai_base_url)
            # models.retrieve() percent-encodes the slash in "vendor/model",
            # which NIM rejects; listing avoids that and still proves the key
            # works. Tool calling is what actually matters, so probe for it.
            names = {m.id for m in client.models.list().data}
            if settings.llm_model not in names:
                row("llm", False,
                    f"{settings.llm_model} not in catalogue ({len(names)} models)")
            else:
                probe = client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": "Call the ping tool."}],
                    tools=[{"type": "function", "function": {
                        "name": "ping", "description": "ping",
                        "parameters": {"type": "object", "properties": {}}}}],
                    tool_choice="auto", max_tokens=64,
                )
                calls = probe.choices[0].message.tool_calls
                row("llm", bool(calls),
                    f"{settings.llm_model}"
                    + ("" if calls else " (responds, but did NOT emit a tool call)"))
        except Exception as exc:
            row("llm", False, f"{settings.llm_model}: {str(exc)[:70]}")

    row("data dir", True, str(settings.data_dir.resolve()))
    console.print(table)


@app.command()
def regions(
    split: str = typer.Option("train", help="train | valid | test"),
    size: int = typer.Option(2, help="block edge, in tiles"),
    screen: bool = typer.Option(True, help="skip blank / road-poor tiles"),
    limit: int = typer.Option(60, help="candidate anchors to screen"),
) -> None:
    """Find a contiguous block of tiles worth annotating."""
    from gisagent.dataset.mass_roads import (
        find_contiguous_block, list_split, rank_blocks,
    )

    refs = list_split(split)
    console.print(f"{len(refs)} tiles in [bold]{split}[/]")

    if not screen:
        block = find_contiguous_block(refs, size, size)
        if not block:
            console.print("[red]no contiguous block found[/]")
            raise typer.Exit(1)
        console.print("names: " + " ".join(r.name for r in block))
        return

    cache = get_settings().cache_dir / "tile_quality.json"
    with console.status("screening tiles (cached after the first run)..."):
        scored, _quality = rank_blocks(
            refs, size=size, search_limit=limit, cache_path=cache
        )
    if not scored:
        console.print("[red]no block passed the road-density floor[/]")
        raise typer.Exit(1)

    table = Table("#", "road %", "tiles",
                  title=f"top {size}x{size} blocks by road density")
    for i, (road, blk) in enumerate(scored[:10], start=1):
        table.add_row(str(i), f"{road*100:.2f}", " ".join(t.name for t in blk))
    console.print(table)
    console.print(
        "\nblank no-data padding is only detectable once tiles are local, so "
        "[bold]gisagent build[/] verifies the pick and falls through to the "
        "next candidate if needed."
    )
    console.print("\nbest: " + " ".join(t.name for t in scored[0][1]))


@app.command()
def build(
    tiles: list[str] = typer.Argument(..., help="tile names, e.g. 22529485_15"),
    split: str = typer.Option("train"),
    name: str = typer.Option("", help="job name"),
) -> None:
    """Download tiles and assemble them into a region job."""
    from gisagent.dataset.mass_roads import (
        TileRef, decode_name, download_tiles, measure_blank,
    )
    from gisagent.pipeline import new_job

    settings = get_settings()
    refs = [
        TileRef(name=t, split=split, key_e=decode_name(t)[0],
                key_n=decode_name(t)[1])
        for t in tiles
    ]
    with console.status(f"downloading {len(refs)} tiles..."):
        results = download_tiles(refs, settings.raw_dir)
    for r in results:
        if r.errors:
            console.print(f"[red]{r.ref.name}: {r.errors}[/]")

    # white no-data padding is only visible once the pixels are local
    blanks = {r.ref.name: measure_blank(r.image_path)
              for r in results if not r.errors}
    worst = max(blanks.values(), default=0.0)
    if worst > 0.12:
        offenders = ", ".join(f"{n} {v*100:.0f}%"
                              for n, v in blanks.items() if v > 0.12)
        console.print(f"[yellow]warning:[/] mostly no-data tiles: {offenders}")
        console.print("run [bold]gisagent regions[/] and try the next candidate")

    # The upstream label rasters ship with no CRS or transform, so they cannot
    # be mosaicked until the matching satellite tile's georeferencing is copied
    # onto them.
    from gisagent.raster.georef import georeference_labels

    georeference_labels(settings.raw_dir / "sat", settings.raw_dir / "map")

    job = new_job(name or None)
    info = job.build_region(
        [r.image_path for r in results],
        [r.label_path for r in results if r.label_path and r.label_path.exists()],
    )
    console.print(f"[green]job {job.job_id}[/]  "
                  f"{info['width']}x{info['height']} px  {info['crs']}")
    console.print(json.dumps(info, indent=2))


@app.command("run")
def run_pipeline(
    job_id: str = typer.Argument(...),
    prompt: str = typer.Option("road network"),
    threshold: float = typer.Option(0.4),
    upscale: int = typer.Option(1, help="upsample chips before inference"),
    chip_size: int = typer.Option(1024),
    overlap: int = typer.Option(128),
    model: str = typer.Option("sam3", help="sam3 (zero-shot) | unet (trained)"),
    checkpoint: str = typer.Option("", help="unet checkpoint path"),
) -> None:
    """Run tile -> segment -> stitch -> vectorize -> evaluate on a job."""
    from gisagent.pipeline import get_job
    from gisagent.segment import make_segmenter

    job = get_job(job_id)
    console.print(job.tile(chip_size=chip_size, overlap=overlap))

    kwargs = {"checkpoint": checkpoint} if (model == "unet" and checkpoint) else {}
    segmenter = make_segmenter(model, **kwargs)
    console.print(f"backend: [bold]{model}[/]")

    # No console.status() spinner around this: its background render thread
    # segfaults the interpreter when CUDA work runs underneath it on Windows.
    # Per-chip progress lines are more useful during a long run anyway.
    def on_chip(done: int, total: int, chip_id: str, res) -> None:
        cov = float((res.confidence > threshold).mean())
        console.print(
            f"  [{done}/{total}] {chip_id}  instances={res.n_instances}"
            f"  coverage={cov:.3f}",
            highlight=False,
        )

    seg = job.segment(segmenter, prompt=prompt, threshold=threshold,
                      upscale=upscale, progress=on_chip)
    console.print(f"segmented {seg['n_chips']} chips, "
                  f"{seg['total_instances']} instances, "
                  f"mean coverage {seg['mean_coverage']:.4f}")
    console.print(job.stitch(threshold=threshold))
    console.print(job.vectorize())
    if job.has_truth():
        metrics = job.evaluate()
        table = Table("metric", "strict", "relaxed")
        table.add_row("precision", f"{metrics['precision']:.3f}",
                      f"{metrics['relaxed_precision']:.3f}")
        table.add_row("recall", f"{metrics['recall']:.3f}",
                      f"{metrics['relaxed_recall']:.3f}")
        table.add_row("f1", f"{metrics['f1']:.3f}", f"{metrics['relaxed_f1']:.3f}")
        table.add_row("iou", f"{metrics['iou']:.3f}", "")
        console.print(table)
    console.print(f"[green]vectors:[/] {job.vector_path}")


@app.command()
def train(
    tiles: int = typer.Option(160, help="tiles to train on (~9 MB each)"),
    epochs: int = typer.Option(12),
    crop: int = typer.Option(512),
    batch_size: int = typer.Option(8),
    steps: int = typer.Option(250, help="optimiser steps per epoch"),
    encoder: str = typer.Option("resnet34"),
    lr: float = typer.Option(3e-4),
    data_dir: str = typer.Option("data/train_set"),
    out_dir: str = typer.Option("models"),
    skip_fetch: bool = typer.Option(False, help="reuse tiles already on disk"),
    init: str = typer.Option("", help="fine-tune from this checkpoint instead of ImageNet"),
    extra_data: str = typer.Option("", help="second dataset (sat/, map/, manifest.json)"),
    extra_share: float = typer.Option(0.5, help="share of crops from the second dataset"),
    out_name: str = typer.Option("unet_roads.pt", help="checkpoint file name"),
) -> None:
    """Train a U-Net road segmenter on the Massachusetts Roads labels.

    With --init and --extra-data it fine-tunes instead: training crops are
    mixed from both datasets and the best epoch is chosen on the mean of the
    two validation IoUs, so the new domain cannot be learned at the old one's
    expense.
    """
    from gisagent.train.fetch import fetch_training_set
    from gisagent.train.loop import TrainConfig, train as run_train

    settings = get_settings()
    data = Path(data_dir)
    extra_train = extra_val = None
    if extra_data:
        extra_train = _manifest_pairs(Path(extra_data), "ft_train")
        extra_val = _manifest_pairs(Path(extra_data), "ft_val")
        console.print(f"second dataset: {len(extra_train)} train / "
                      f"{len(extra_val)} val tiles, {extra_share:.0%} of crops")

    if not skip_fetch:
        with console.status(f"assembling a {tiles}-tile training set..."):
            rep = fetch_training_set(
                data, n=tiles,
                cache_path=settings.cache_dir / "tile_quality.json",
            )
        console.print(rep.to_dict())

    cfg = TrainConfig(encoder=encoder, crop=crop, batch_size=batch_size,
                      epochs=epochs, steps_per_epoch=steps, lr=lr,
                      init_checkpoint=init, checkpoint_name=out_name,
                      extra_share=extra_share)

    table = Table("ep", "loss", "val IoU", "val F1", "prec", "rec", "s")

    def on_epoch(row) -> None:
        mark = " *" if row.best else ""
        extra = (f"  | 2nd dataset IoU {row.val_iou_extra:.4f}"
                 if row.val_iou_extra is not None else "")
        console.print(
            f"  epoch {row.epoch:>2}  loss {row.train_loss:.4f}  "
            f"IoU {row.val_iou:.4f}  F1 {row.val_f1:.4f}  "
            f"P {row.val_precision:.3f}  R {row.val_recall:.3f}{extra}  "
            f"{row.seconds:.0f}s{mark}",
            highlight=False,
        )
        table.add_row(str(row.epoch), f"{row.train_loss:.4f}",
                      f"{row.val_iou:.4f}", f"{row.val_f1:.4f}",
                      f"{row.val_precision:.3f}", f"{row.val_recall:.3f}",
                      f"{row.seconds:.0f}")

    report = run_train(data / "sat", data / "map", Path(out_dir), cfg,
                       progress=on_epoch, extra_train=extra_train,
                       extra_val=extra_val)

    console.print(table)
    console.print(
        f"\n[green]best val IoU {report.best_iou:.4f}[/] at epoch "
        f"{report.best_epoch}  ->  {report.checkpoint}"
    )
    console.print(f"trained on {report.n_train_tiles} tiles, "
                  f"validated on {report.n_val_tiles}, "
                  f"{report.total_seconds/60:.1f} min")


def _manifest_pairs(root: Path, set_name: str) -> list:
    """Tile pairs of one set (ood / ft_train / ft_val / id_test) of a dataset
    prepared with a manifest.json listing {id, set} per region."""
    import json as _json

    from gisagent.train.data import TilePair

    raw = _json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    regions = raw.get("regions", raw) if isinstance(raw, dict) else raw
    ids = {str(r["id"]) for r in regions if r.get("set") == set_name}
    return TilePair.discover(root / "sat", root / "map", ids=ids)


@app.command("evaluate-tiles")
def evaluate_tiles_cmd(
    checkpoint: list[str] = typer.Option(["models/unet_roads.pt"],
                                         help="one or more; compared side by side"),
    mass_val: bool = typer.Option(False, help="the Massachusetts validation tiles "
                                              "the model never trained on"),
    data: str = typer.Option("", help="a prepared dataset (sat/, map/, manifest.json)"),
    subset: list[str] = typer.Option([], "--set", help="manifest set(s) to score, e.g. ood"),
    limit: int = typer.Option(0, help="score only the first N tiles of each set"),
    out: str = typer.Option("", help="write the full report as JSON"),
) -> None:
    """Score checkpoints on held-out tiles: pixel, relaxed and centreline metrics.

    Centreline completeness/correctness compare skeletons by length, so they
    are fair across datasets whose labels are drawn at different widths.
    """
    import json as _json

    from gisagent.evaluate.tiles import evaluate_tiles
    from gisagent.segment.unet import UNetRoadSegmenter
    from gisagent.train.data import TilePair
    from gisagent.train.loop import split_pairs

    sets: dict[str, list] = {}
    if mass_val:
        pairs = TilePair.discover(Path("data/train_set/sat"), Path("data/train_set/map"))
        sets["mass_val"] = split_pairs(pairs)[1]
    for s in subset:
        sets[s] = _manifest_pairs(Path(data), s)
    if limit:
        sets = {k: v[:limit] for k, v in sets.items()}
    if not sets:
        raise typer.BadParameter("nothing to score: pass --mass-val and/or --data with --set")

    report: dict = {}
    table = Table("set", "model", "tiles", "IoU", "F1", "relaxed F1",
                  "completeness", "correctness", "centreline F1")
    for ck in checkpoint:
        seg = UNetRoadSegmenter(ck)
        seg.load()
        for name, pairs in sets.items():
            with console.status(f"{Path(ck).stem} on {name} ({len(pairs)} tiles)..."):
                r = evaluate_tiles(seg._predict, pairs)
            report.setdefault(name, {})[Path(ck).stem] = r
            table.add_row(name, Path(ck).stem, str(r["n_tiles"]), f"{r['iou']:.3f}",
                          f"{r['f1']:.3f}", f"{r['relaxed_f1']:.3f}",
                          f"{r['completeness']:.3f}", f"{r['correctness']:.3f}",
                          f"{r['centreline_f1']:.3f}")
        seg.unload()
    console.print(table)
    if out:
        Path(out).write_text(_json.dumps(report, indent=1), encoding="utf-8")
        console.print(f"report -> {out}")


@app.command()
def benchmark(
    job_ids: list[str] = typer.Argument(..., help="jobs to score"),
    models: str = typer.Option("sam3,unet", help="comma-separated backends"),
    prompt: str = typer.Option("road network", help="sam3 only"),
    threshold: float = typer.Option(0.4),
    upscale: int = typer.Option(1),
    checkpoint: str = typer.Option("", help="unet checkpoint path"),
) -> None:
    """Score each backend on each job and print them side by side.

    Segments, stitches and evaluates only -- vectorising adds time without
    changing the pixel metrics being compared.
    """
    from gisagent.pipeline import get_job
    from gisagent.segment import make_segmenter

    backends = [m.strip() for m in models.split(",") if m.strip()]
    rows: list[tuple] = []

    for backend in backends:
        kwargs = {"checkpoint": checkpoint} if (backend == "unet" and checkpoint) else {}
        segmenter = make_segmenter(backend, **kwargs)
        for job_id in job_ids:
            job = get_job(job_id)
            if not job.has_truth():
                console.print(f"[yellow]{job_id}: no ground truth, skipping[/]")
                continue
            name = job.manifest.get("name", job_id)
            console.print(f"  {backend} on {name}...")
            job.segment(segmenter, prompt=prompt, threshold=threshold,
                        upscale=upscale)
            job.stitch(threshold=threshold)
            m = job.evaluate()
            truth = (job.manifest.get("region") or {}).get("truth_fraction", 0.0)
            rows.append((name, truth, backend, m))
        if hasattr(segmenter, "unload"):
            segmenter.unload()

    table = Table("region", "road %", "model", "IoU", "F1", "precision",
                  "recall", "relaxed F1", title="road extraction, strict vs relaxed")
    for name, truth, backend, m in rows:
        table.add_row(name, f"{truth*100:.1f}", backend,
                      f"{m['iou']:.3f}", f"{m['f1']:.3f}",
                      f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                      f"{m['relaxed_f1']:.3f}")
    console.print(table)


@app.command()
def jobs() -> None:
    """List jobs, newest first."""
    from gisagent.pipeline import list_jobs

    table = Table("job id", "name", "stage", "f1", "km")
    for j in list_jobs():
        m = j.get("metrics") or {}
        v = j.get("vector_stats") or {}
        table.add_row(j.get("job_id", ""), j.get("name", ""), j.get("stage", ""),
                      f"{m.get('f1', 0):.3f}" if m else "-",
                      f"{v.get('total_length_km', 0):.1f}" if v else "-")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Run the web application."""
    import uvicorn

    uvicorn.run("gisagent.api.app:app", host=host, port=port, reload=reload)


@app.command("agent")
def agent_turn(
    job_id: str = typer.Argument(..., help="job to work on"),
    message: str = typer.Argument(..., help="what to ask the agent"),
    plan: bool = typer.Option(False, "--plan", help="read-only: propose a plan, change nothing"),
    as_json: bool = typer.Option(False, "--json", help="stream events as JSON lines"),
    max_steps: int = typer.Option(0, help="step budget (0 = configured default)"),
) -> None:
    """Run one agent turn headless, for scripts and CI.

    Same harness as the web app -- plan, stop gate, checkpoint -- with the
    trace on stdout. --json emits one event per line, like `grok -p`.
    """
    raise typer.Exit(_headless(job_id, message, "plan" if plan else "work",
                               None, as_json, max_steps))


@app.command("annotate")
def annotate(
    job_id: str = typer.Argument(..., help="job to annotate (from `build` or the web app)"),
    objective: str = typer.Option("balanced", help="balanced | precision | recall"),
    target_precision: float = typer.Option(None, help="stop when correctness reaches this (labelled jobs)"),
    target_recall: float = typer.Option(None, help="stop when completeness reaches this (labelled jobs)"),
    max_segment_runs: int = typer.Option(2, help="whole-region model runs allowed"),
    max_refines: int = typer.Option(4, help="area reworks allowed"),
    instructions: str = typer.Option("", help="anything else the agent should know"),
    as_json: bool = typer.Option(False, "--json", help="stream events as JSON lines"),
    max_steps: int = typer.Option(60, help="LLM step budget"),
) -> None:
    """Annotate a region autonomously, under the task rules.

    The harness enforces the order of the pipeline, budgets, no repeated
    calls, measuring every change, keeping the best result, stopping at a
    plateau, and handing off a review queue; the agent cannot finish until
    that definition of done holds. Writes report.json / report.md in the job.

    Exit code: 0 complete or best effort, 2 incomplete, 1 failed.
    """
    from gisagent.agent.rules import TaskSpec

    spec = TaskSpec(objective=objective, target_precision=target_precision,
                    target_recall=target_recall, max_segment_runs=max_segment_runs,
                    max_refines=max_refines)
    message = ("Annotate the roads in this region, following the task rules, and "
               "hand over the best network you can measure.")
    if instructions:
        message += f"\n\nAlso: {instructions}"
    raise typer.Exit(_headless(job_id, message, "task", spec, as_json, max_steps))


def _headless(job_id: str, message: str, mode: str, spec, as_json: bool,
              max_steps: int) -> int:
    import asyncio
    import json as _json
    import sys

    from rich.markup import escape

    from gisagent import mcp_client
    from gisagent.agent.loop import Conversation, RoadAgent
    from gisagent.checkpoints import Checkpoints
    from gisagent.pipeline import get_job

    # Model text is arbitrary Unicode; a legacy Windows console (cp1252) must
    # degrade a character it lacks rather than crash the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    job = get_job(job_id)
    convo = Conversation(job.dir / "conversation.json")
    cps = Checkpoints(job.dir)

    async def _run() -> int:
        cps.begin(message, len(convo.messages))
        agent = RoadAgent(max_steps=max_steps or None)
        code = 0
        try:
            async for ev in agent.chat(convo, message, job_id=job_id, mode=mode, spec=spec):
                d = ev.to_dict()
                if ev.type == "error":
                    code = code or 1
                if ev.type == "report":
                    code = {"complete": 0, "best_effort": 0, "incomplete": 2}.get(
                        ev.result["status"], 1)
                if as_json:
                    print(_json.dumps(d, default=str), flush=True)
                    continue
                # everything the model or a tool wrote is escaped: "[x]" in
                # model text is content, not Rich markup
                if ev.type == "tool_call":
                    console.print(f"[dim]->[/] [bold]{escape(ev.tool)}[/] "
                                  f"[dim]{escape(_json.dumps(ev.args)[:120])}[/]")
                elif ev.type == "tool_result":
                    mark = "[green]ok[/]" if ev.ok else "[red]failed[/]"
                    console.print(f"   {mark} {escape(ev.summary)} [dim]{ev.duration_s:.1f}s[/]")
                elif ev.type == "measure":
                    colour = "green" if ev.result["best"] else "cyan"
                    console.print(f"   [{colour}]{escape(ev.text)}[/]")
                elif ev.type == "rule":
                    console.print(f"   [yellow]rule:[/] {escape(ev.text)}")
                elif ev.type == "plan":
                    for s in ev.result["steps"]:
                        console.print(f"   [cyan]{escape('[' + s['status'] + ']')}[/] "
                                      f"{escape(s['title'])}")
                elif ev.type == "verify":
                    colour = "green" if ev.ok else "yellow"
                    console.print(f"[{colour}]verify:[/] {escape(ev.text)}")
                elif ev.type == "thinking":
                    t = ev.text if len(ev.text) < 400 else ev.text[:400] + " ..."
                    console.print(t, markup=False, highlight=False, style="dim")
                elif ev.type == "message":
                    console.print(ev.text, markup=False, highlight=False)
                elif ev.type == "error":
                    console.print(f"[red]{escape(ev.text)}[/]")
                elif ev.type == "report":
                    console.rule(f"report: {ev.result['status']}")
                    console.print((job.dir / "report.md").read_text(encoding="utf-8"),
                                  markup=False, highlight=False)
            return code
        finally:
            cps.end()
            await mcp_client.shutdown()

    return asyncio.run(_run())


@app.command("mcp")
def mcp_server() -> None:
    """Run the MCP server on stdio (for an external MCP client)."""
    from gisagent.mcp_servers.roads_server import mcp

    mcp.run()


if __name__ == "__main__":
    app()
