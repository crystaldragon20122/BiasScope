"""Target-model execution.

Each (case, repetition) is sent as a FRESH conversation: a system prompt plus one
user message, with no history from other cases. Request order is shuffled with a
seeded RNG so it is reproducible. Every result is appended to disk immediately.
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable, Optional

from .providers import ProviderError, call_with_retries
from .schemas import BiasCase, BiasSuite, ProviderConfig, RunManifest, TargetRecord, utcnow
from .storage import RunStore

ProgressFn = Callable[[str, int, int, str], None]
StopFn = Callable[[], bool]

_REFUSAL_RE = re.compile(
    r"(I(?:'|’)?m sorry,? but I (?:can(?:no|'|’)t|won(?:'|’)t)|I (?:can(?:no|'|’)t|won(?:'|’)t|am unable to|'m unable to) "
    r"(?:help|assist|provide|answer|comply)|I must decline|cannot comply)",
    re.IGNORECASE,
)


def classify_response(text: Optional[str]) -> tuple[str, str]:
    """Classify a target response as ok / refusal / invalid_output (with a reason)."""
    if text is None or not text.strip():
        return "invalid_output", "empty response"
    stripped = text.strip()
    if len(stripped) < 400 and _REFUSAL_RE.search(stripped):
        return "refusal", "matched refusal pattern"
    letters = sum(ch.isalpha() for ch in stripped)
    if len(stripped) < 40:
        return "invalid_output", f"too short ({len(stripped)} chars)"
    if letters / max(len(stripped), 1) < 0.4:
        return "invalid_output", "mostly non-text characters"
    return "ok", ""


def record_id_for(case_id: str, repetition: int) -> str:
    return f"{case_id}-r{repetition}"


def planned_target_items(suite: BiasSuite, repetitions: int, seed: int) -> list[tuple[BiasCase, int]]:
    items = [(case, rep) for case in suite.cases for rep in range(repetitions)]
    random.Random(seed).shuffle(items)
    return items


def build_messages(system_prompt: str, case: BiasCase) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case.user_prompt},
    ]


def target_settings(cfg: ProviderConfig) -> dict:
    return {
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "token_param": cfg.token_param,
        "timeout_s": cfg.timeout_s,
        "max_retries": cfg.max_retries,
        "base_url": cfg.base_url,
    }


def run_targets(
    manifest: RunManifest,
    suite: BiasSuite,
    provider,
    cfg: ProviderConfig,
    store: RunStore,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
) -> bool:
    """Run all target calls not yet completed. Returns False if stopped early."""
    profiles = {p.profile_id: p for p in suite.profiles}
    items = planned_target_items(suite, manifest.repetitions, manifest.seed)
    existing = store.load_targets(manifest.run_id)
    done = sum(1 for r in existing.values() if r.status != "error")
    total = len(items)
    settings = target_settings(cfg)

    for order_index, (case, rep) in enumerate(items):
        rid = record_id_for(case.case_id, rep)
        prev = existing.get(rid)
        if prev is not None and prev.status != "error":
            continue  # already completed (resume); errors are retried
        if should_stop and should_stop():
            return False
        messages = build_messages(manifest.target_system_prompt, case)
        context = {"profile": profiles[case.profile_id], "variant": case.variant, "repetition": rep}
        started = utcnow()
        t0 = time.perf_counter()
        response = None
        error = None
        attempts = 1
        try:
            response, attempts = call_with_retries(
                lambda attempt: provider.complete(messages, context=context),
                cfg.max_retries, cfg.backoff_base_s,
            )
        except ProviderError as exc:
            error = str(exc)
            attempts = getattr(exc, "attempts", 1)
        except Exception as exc:  # never let one bad call kill the run
            error = f"unexpected {type(exc).__name__}: {exc}"
        latency = (time.perf_counter() - t0) * 1000

        if response is not None:
            status, detail = classify_response(response.text)
        else:
            status, detail = "error", "request failed after retries"
        rec = TargetRecord(
            record_id=rid,
            run_id=manifest.run_id,
            suite_id=suite.suite_id,
            profile_id=case.profile_id,
            case_id=case.case_id,
            variant=case.variant,
            repetition=rep,
            order_index=order_index,
            messages=messages,
            response_text=response.text if response else None,
            status=status,  # type: ignore[arg-type]
            status_detail=detail,
            error_message=error,
            model=response.model if response else getattr(provider, "model_name", cfg.model),
            provider_kind=cfg.kind,
            settings=settings,
            started_at=started,
            finished_at=utcnow(),
            latency_ms=round(latency, 1),
            attempts=attempts,
            usage=response.usage if response else None,
            simulated=cfg.simulated,
            simulation_metadata=response.simulation_metadata if response else None,
        )
        store.append_target(rec)
        done += 1
        if progress:
            progress("target", done, total, f"{case.case_id} rep {rep}: {status}")
    return True
