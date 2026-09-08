"""Guards for the CLI's lazy imports.

Every command imports its heavy dependencies inside the function body, to keep
`--help` fast and to avoid loading torch for commands that never touch it. The
cost is that a renamed symbol is invisible until someone runs that exact
command -- which is how `Sam3Segmenter`, a `version()` that was really a
method, and a missing georeferencing step all shipped. These tests resolve the
names without executing the work.
"""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

from gisagent.cli import app

runner = CliRunner()

# (module, attribute) pairs the CLI resolves at call time
LAZY_IMPORTS = [
    ("gisagent.pipeline", "get_job"),
    ("gisagent.pipeline", "new_job"),
    ("gisagent.pipeline", "list_jobs"),
    ("gisagent.segment.sam3", "Sam3RoadSegmenter"),
    ("gisagent.qgis.process", "QgisProcess"),
    ("gisagent.qgis.process", "available"),
    ("gisagent.raster.georef", "georeference_labels"),
    ("gisagent.dataset.mass_roads", "list_split"),
    ("gisagent.dataset.mass_roads", "rank_blocks"),
    ("gisagent.dataset.mass_roads", "find_contiguous_block"),
    ("gisagent.dataset.mass_roads", "download_tiles"),
    ("gisagent.dataset.mass_roads", "measure_blank"),
    ("gisagent.dataset.mass_roads", "decode_name"),
    ("gisagent.dataset.mass_roads", "TileRef"),
    ("gisagent.mcp_servers.roads_server", "mcp"),
]


@pytest.mark.parametrize("module,attr", LAZY_IMPORTS)
def test_lazily_imported_symbol_exists(module, attr):
    assert hasattr(importlib.import_module(module), attr), (
        f"{module}.{attr} is referenced by the CLI but does not exist"
    )


def test_qgis_version_is_a_method_not_a_module_function():
    """It reads like a module function; it is not. This caught a real bug."""
    from gisagent.qgis import process

    assert not hasattr(process, "version")
    assert callable(process.QgisProcess.version)


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "regions", "build", "run", "jobs", "serve", "mcp"):
        assert command in result.stdout


@pytest.mark.parametrize(
    "command", ["doctor", "regions", "build", "run", "jobs", "serve", "mcp"]
)
def test_each_command_has_help(command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


def test_tileref_is_constructible_the_way_the_cli_builds_it():
    """`TileRef(t, split, *decode_name(t))` raised TypeError: name given twice."""
    from gisagent.dataset.mass_roads import TileRef, decode_name

    name = "22529485_15"
    key_e, key_n = decode_name(name)
    ref = TileRef(name=name, split="train", key_e=key_e, key_n=key_n)
    assert ref.name == name
    assert ref.sat_url.endswith(".tiff")
    assert ref.map_url.endswith(".tif")
