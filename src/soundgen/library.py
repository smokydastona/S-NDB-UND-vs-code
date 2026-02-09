from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class CatalogRecord:
    created_utc: str
    engine: str
    prompt: str
    namespace: str
    event: str
    sound_path: str
    output_file: str
    tags: tuple[str, ...] = ()

    seconds: Optional[float] = None
    seed: Optional[int] = None
    preset: Optional[str] = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_record(catalog_path: Path, record: CatalogRecord) -> None:
    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def make_record(
    *,
    engine: str,
    prompt: str,
    namespace: str,
    event: str,
    sound_path: str,
    output_file: Path,
    tags: tuple[str, ...] = (),
    seconds: Optional[float] = None,
    seed: Optional[int] = None,
    preset: Optional[str] = None,
) -> CatalogRecord:
    return CatalogRecord(
        created_utc=_now_utc_iso(),
        engine=str(engine),
        prompt=str(prompt),
        namespace=str(namespace),
        event=str(event),
        sound_path=str(sound_path),
        output_file=str(output_file),
        tags=tuple(tags),
        seconds=seconds,
        seed=seed,
        preset=preset,
    )
