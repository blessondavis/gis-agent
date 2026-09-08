# gis-agent

Agentic road annotation on satellite imagery.

Feeds georeferenced satellite tiles to a segmentation model (SAM 3), has an LLM
agent drive the pipeline over MCP tools, vectorizes the result to road
centrelines with headless QGIS/GDAL, and serves the whole thing as a web app
with a live view of what the agent is doing.

See `docs/` for architecture and `.env.example` for configuration.
