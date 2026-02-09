from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class ReplicateParams:
    prompt: str
    seconds: float
    out_path: Path

    model: str
    api_token: Optional[str] = None

    extra_input_json: Optional[str] = None

    poll_interval_s: float = 1.0
    timeout_s: float = 300.0


def _token(params: ReplicateParams) -> str:
    tok = params.api_token or os.getenv("REPLICATE_API_TOKEN")
    if not tok:
        raise RuntimeError(
            "Missing Replicate API token. Set REPLICATE_API_TOKEN env var or pass --replicate-token."
        )
    return tok


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return dest


def generate_with_replicate(params: ReplicateParams) -> Path:
    """Run a Replicate prediction and download the resulting audio.

    This is intentionally generic: different models expect different input schemas.
    We send {prompt, duration} plus any JSON provided via extra_input_json.
    """

    headers = {
        "Authorization": f"Token {_token(params)}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    model = params.model.strip()
    if not model:
        raise ValueError("replicate model is required")

    input_obj: dict[str, Any] = {
        "prompt": params.prompt,
        "duration": float(params.seconds),
        "seconds": float(params.seconds),
    }

    if params.extra_input_json:
        extra = json.loads(params.extra_input_json)
        if not isinstance(extra, dict):
            raise ValueError("--replicate-input-json must parse to a JSON object")
        input_obj.update(extra)

    body = {"model": model, "input": input_obj}

    create = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        data=json.dumps(body),
        timeout=60,
    )
    create.raise_for_status()
    pred = create.json()
    pred_url = pred.get("urls", {}).get("get")
    if not isinstance(pred_url, str):
        raise RuntimeError("Replicate response missing prediction URL")

    deadline = time.time() + float(params.timeout_s)
    while True:
        if time.time() > deadline:
            raise TimeoutError("Replicate prediction timed out")

        r = requests.get(pred_url, headers=headers, timeout=60)
        r.raise_for_status()
        pred = r.json()
        status = pred.get("status")

        if status in {"failed", "canceled"}:
            err = pred.get("error")
            raise RuntimeError(f"Replicate prediction {status}: {err}")

        if status == "succeeded":
            output = pred.get("output")
            # output can be URL string or list of URLs
            if isinstance(output, str):
                url = output
            elif isinstance(output, list) and output and isinstance(output[0], str):
                url = output[0]
            else:
                raise RuntimeError("Replicate prediction succeeded but output was not a URL")

            # Choose extension from URL if possible
            ext = Path(url.split("?")[0]).suffix or ".wav"
            out = params.out_path
            if out.suffix.lower() == ".wav" and ext.lower() != ".wav":
                # keep user's chosen extension for now
                pass
            elif out.suffix == "":
                out = out.with_suffix(ext)

            return _download(url, out)

        time.sleep(float(params.poll_interval_s))
