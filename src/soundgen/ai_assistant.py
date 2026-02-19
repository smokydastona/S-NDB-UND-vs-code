from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import requests


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class AIChatError(RuntimeError):
    pass


def default_system_prompt() -> str:
    return (
        "You are SÖNDBÖUND Copilot, an assistant for generating Minecraft-ready SFX and asset-pack metadata. "
        "Be concise and practical. When the user asks for structured output (like a batch manifest, sounds.json entries, "
        "or subtitle lang keys), output valid JSON and nothing else. "
        "Assume audio generation pipelines are deterministic and should not be changed unless the user asks. "
        "You can help with prompt writing, batch manifests, naming sound events, creating subtitles, explaining engines, "
        "and explaining errors."
    )


def _safe_snip(text: str, limit: int = 800) -> str:
    t = str(text or "")
    t = t.replace("\r\n", "\n")
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def _http_error(prefix: str, resp: requests.Response) -> AIChatError:
    body = ""
    try:
        body = resp.text or ""
    except Exception:
        body = ""
    msg = f"{prefix}: HTTP {resp.status_code} {resp.reason}. { _safe_snip(body) }"
    return AIChatError(msg)


def _as_messages(system_prompt: str, history: list[tuple[str, str]] | None, user_text: str) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    sys_p = str(system_prompt or "").strip()
    if sys_p:
        msgs.append({"role": "system", "content": sys_p})

    for u, a in (history or []):
        uu = str(u or "").strip()
        aa = str(a or "").strip()
        if uu:
            msgs.append({"role": "user", "content": uu})
        if aa:
            msgs.append({"role": "assistant", "content": aa})

    ut = str(user_text or "").strip()
    if ut:
        msgs.append({"role": "user", "content": ut})
    return msgs


def chat_ollama(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout_s: float = 60.0,
) -> str:
    base = str(base_url or "").strip() or "http://localhost:11434"
    url = base.rstrip("/") + "/api/chat"

    payload: dict[str, Any] = {
        "model": str(model or "").strip() or "llama3.2",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except Exception as e:
        raise AIChatError(f"Ollama request failed: {e}") from e

    if resp.status_code >= 400:
        raise _http_error("Ollama request failed", resp)

    try:
        data = resp.json()
    except Exception as e:
        raise AIChatError(f"Ollama returned non-JSON: {e}. { _safe_snip(resp.text) }") from e

    content = (((data or {}).get("message") or {}).get("content") or "")
    out = str(content).strip()
    if not out:
        raise AIChatError(f"Ollama returned empty response. { _safe_snip(json.dumps(data, ensure_ascii=False)) }")
    return out


def ollama_reachable(*, base_url: str, timeout_s: float = 2.5) -> bool:
    base = str(base_url or "").strip() or "http://localhost:11434"
    url = base.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout_s)
        return resp.status_code < 400
    except Exception:
        return False


def ollama_list_models(*, base_url: str, timeout_s: float = 5.0) -> list[str]:
    base = str(base_url or "").strip() or "http://localhost:11434"
    url = base.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout_s)
    except Exception as e:
        raise AIChatError(f"Ollama not reachable at {base}: {e}") from e
    if resp.status_code >= 400:
        raise _http_error("Ollama tags request failed", resp)
    try:
        data = resp.json()
    except Exception as e:
        raise AIChatError(f"Ollama tags returned non-JSON: {e}. { _safe_snip(resp.text) }") from e
    models = []
    for m in (data or {}).get("models") or []:
        if isinstance(m, dict) and m.get("name"):
            models.append(str(m["name"]))
    return models


def ollama_pull_model(
    *,
    base_url: str,
    model: str,
    timeout_s: float = 0.0,
):
    """Yield (pct, status) updates while pulling a model.

    Uses Ollama's streaming JSON-lines response.
    """

    base = str(base_url or "").strip() or "http://localhost:11434"
    url = base.rstrip("/") + "/api/pull"
    payload: dict[str, Any] = {
        "name": str(model or "").strip() or "llama3.2",
        "stream": True,
    }

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=(10.0, None if timeout_s <= 0 else timeout_s))
    except Exception as e:
        raise AIChatError(f"Ollama pull request failed: {e}") from e

    if resp.status_code >= 400:
        raise _http_error("Ollama pull request failed", resp)

    last_status = "pulling"
    last_pct = 0
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                j = json.loads(raw)
            except Exception:
                continue
            status = str(j.get("status") or last_status).strip() or last_status
            last_status = status

            completed = j.get("completed")
            total = j.get("total")
            pct = None
            try:
                if completed is not None and total:
                    pct = int(round((float(completed) / float(total)) * 100.0))
            except Exception:
                pct = None

            if pct is None:
                pct = last_pct
            pct = max(0, min(100, int(pct)))
            last_pct = pct
            yield pct, status
    finally:
        try:
            resp.close()
        except Exception:
            pass


def chat_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout_s: float = 60.0,
) -> str:
    base = str(base_url or "").strip() or "https://api.openai.com/v1"
    base = base.rstrip("/")

    if base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"

    key = str(api_key or "").strip()
    if not key:
        raise AIChatError("Missing API key for OpenAI-compatible provider.")

    payload: dict[str, Any] = {
        "model": str(model or "").strip() or "gpt-4o-mini",
        "messages": messages,
        "temperature": float(temperature),
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    except Exception as e:
        raise AIChatError(f"OpenAI-compatible request failed: {e}") from e

    if resp.status_code >= 400:
        raise _http_error("OpenAI-compatible request failed", resp)

    try:
        data = resp.json()
    except Exception as e:
        raise AIChatError(f"OpenAI-compatible returned non-JSON: {e}. { _safe_snip(resp.text) }") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise AIChatError(
            f"OpenAI-compatible response schema unexpected: {e}. { _safe_snip(json.dumps(data, ensure_ascii=False)) }"
        ) from e

    out = str(content or "").strip()
    if not out:
        raise AIChatError("OpenAI-compatible returned empty response.")
    return out


def chat_azure_openai(
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout_s: float = 60.0,
) -> str:
    base = str(endpoint or "").strip()
    if not base:
        raise AIChatError("Missing Azure OpenAI endpoint.")
    base = base.rstrip("/")

    dep = str(deployment or "").strip()
    if not dep:
        raise AIChatError("Missing Azure OpenAI deployment name.")

    key = str(api_key or "").strip()
    if not key:
        raise AIChatError("Missing API key for Azure OpenAI.")

    ver = str(api_version or "").strip() or "2024-02-15-preview"

    url = f"{base}/openai/deployments/{dep}/chat/completions"
    params = {"api-version": ver}

    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": float(temperature),
    }

    headers = {
        "api-key": key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, params=params, json=payload, headers=headers, timeout=timeout_s)
    except Exception as e:
        raise AIChatError(f"Azure OpenAI request failed: {e}") from e

    if resp.status_code >= 400:
        raise _http_error("Azure OpenAI request failed", resp)

    try:
        data = resp.json()
    except Exception as e:
        raise AIChatError(f"Azure OpenAI returned non-JSON: {e}. { _safe_snip(resp.text) }") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise AIChatError(
            f"Azure OpenAI response schema unexpected: {e}. { _safe_snip(json.dumps(data, ensure_ascii=False)) }"
        ) from e

    out = str(content or "").strip()
    if not out:
        raise AIChatError("Azure OpenAI returned empty response.")
    return out


ProviderKind = Literal["local-ollama", "cloud-openai", "cloud-azure"]


def chat_once(
    *,
    provider: ProviderKind,
    user_text: str,
    history: list[tuple[str, str]] | None,
    system_prompt: str | None = None,
    endpoint: str | None = None,
    model_or_deployment: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
) -> str:
    sys_p = (system_prompt or os.environ.get("SOUNDGEN_AI_SYSTEM_PROMPT") or "").strip() or default_system_prompt()
    messages = _as_messages(sys_p, history, user_text)

    if provider == "local-ollama":
        return chat_ollama(
            base_url=str(endpoint or "http://localhost:11434"),
            model=str(model_or_deployment or "llama3.2"),
            messages=messages,
            temperature=float(temperature),
            timeout_s=float(timeout_s),
        )

    if provider == "cloud-azure":
        return chat_azure_openai(
            endpoint=str(endpoint or "").strip(),
            api_key=str(api_key or "").strip(),
            deployment=str(model_or_deployment or "").strip(),
            api_version=str(api_version or "").strip(),
            messages=messages,
            temperature=float(temperature),
            timeout_s=float(timeout_s),
        )

    # cloud-openai
    return chat_openai_compatible(
        base_url=str(endpoint or "https://api.openai.com/v1"),
        api_key=str(api_key or "").strip(),
        model=str(model_or_deployment or "gpt-4o-mini"),
        messages=messages,
        temperature=float(temperature),
        timeout_s=float(timeout_s),
    )
