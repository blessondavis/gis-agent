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
) -> None:
    """Run tile -> segment -> stitch -> vectorize -> evaluate on a job."""
    from gisagent.pipeline import get_job
    from gisagent.segment.sam3 import Sam3RoadSegmenter

    job = get_job(job_id)
    console.print(job.tile(chip_size=chip_size, overlap=overlap))

    segmenter = Sam3RoadSegmenter()

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


@app.command("mcp")
def mcp_server() -> None:
    """Run the MCP server on stdio (for an external MCP client)."""
    from gisagent.mcp_servers.roads_server import mcp

    mcp.run()


if __name__ == "__main__":
    app()
