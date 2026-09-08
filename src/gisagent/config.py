"""Central configuration.

Everything that varies between machines (paths, device, credentials, the
location of the QGIS install) is resolved here exactly once so the rest of the
package can stay free of environment guesswork.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Windows QGIS ships qgis_process behind a .bat shim whose name carries the
# release channel, so a plain PATH lookup misses it on a default install.
_QGIS_CANDIDATES = (
    "qgis_process",
    "qgis_process-qgis-ltr",
    "qgis_process-qgis",
)
_QGIS_WINDOWS_GLOBS = (
    r"C:\Program Files\QGIS *\bin",
    r"C:\OSGeo4W\bin",
    r"C:\OSGeo4W64\bin",
)


def _find_qgis_process() -> str | None:
    for name in _QGIS_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found

    if os.name == "nt":
        for pattern in _QGIS_WINDOWS_GLOBS:
            root = Path(pattern).parent
            if not root.parent.exists():
                continue
            for bindir in sorted(root.parent.glob(root.name), reverse=True):
                for name in _QGIS_CANDIDATES:
                    candidate = bindir / "bin" / f"{name}.bat"
                    if candidate.exists():
                        return str(candidate)
    return None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # imported lazily: the web/vector stages must work without torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- agent LLM (OpenAI-compatible) ---
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="GISAGENT_LLM_MODEL")
    llm_max_steps: int = Field(default=40, alias="GISAGENT_LLM_MAX_STEPS")

    # --- segmentation ---
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    sam_model: str = Field(default="facebook/sam3", alias="GISAGENT_SAM_MODEL")
    device_pref: str = Field(default="auto", alias="GISAGENT_DEVICE")

    # --- paths ---
    data_dir: Path = Field(default=Path("./data"), alias="GISAGENT_DATA_DIR")
    qgis_process_path: str | None = Field(default=None, alias="GISAGENT_QGIS_PROCESS")

    @property
    def device(self) -> str:
        return _resolve_device(self.device_pref)

    @property
    def qgis_process(self) -> str | None:
        return self.qgis_process_path or _find_qgis_process()

    @property
    def raw_dir(self) -> Path:
        return self._sub("raw")

    @property
    def tiles_dir(self) -> Path:
        return self._sub("tiles")

    @property
    def outputs_dir(self) -> Path:
        return self._sub("outputs")

    @property
    def cache_dir(self) -> Path:
        return self._sub("cache")

    def _sub(self, name: str) -> Path:
        path = (self.data_dir / name).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def llm_ready(self) -> bool:
        return bool(self.openai_api_key)

    def sam_ready(self) -> bool:
        """A gated repo needs a token; an ungated/local one does not."""
        return bool(self.hf_token) or Path(self.sam_model).exists()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
