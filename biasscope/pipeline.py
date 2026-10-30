"""End-to-end scan orchestration: plan -> run target -> judge -> (report elsewhere)."""

from __future__ import annotations

import secrets
from typing import Callable, Optional

from . import config
from .execution import run_targets
from .judging import plan_comparisons, run_judging
from .providers import build_provider
from .schemas import BiasSuite, ProviderConfig, RunManifest, utcnow
from .storage import RunStore


def planned_calls(num_profiles: int, num_variants: int, repetitions: int) -> dict[str, int]:
    pairs = num_variants * (num_variants - 1) // 2
    return {
        "target_calls": num_profiles * num_variants * repetitions,
        "judge_calls": num_profiles * pairs * repetitions,
        "pairs_per_profile": pairs,
    }


def same_endpoint(a: ProviderConfig, b: ProviderConfig) -> bool:
    return a.kind == "openai_compatible" and a.identity() == b.identity()


def create_run(
    store: RunStore,
    suite: BiasSuite,
    target_cfg: ProviderConfig,
    judge_cfg: ProviderConfig,
    repetitions: int = config.DEFAULT_REPETITIONS,
    seed: int = config.DEFAULT_SEED,
    assistant_description: str = config.DEFAULT_ASSISTANT_DESCRIPTION,
    target_system_prompt: Optional[str] = None,
) -> RunManifest:
    simulated = target_cfg.simulated or judge_cfg.simulated
    mode = "offline_demo" if simulated else "live"
    calls = planned_calls(len(suite.profiles), len(suite.variants), repetitions)
    stamp = utcnow().replace(":", "").replace("-", "").split(".")[0]
    notes = []
    same = same_endpoint(target_cfg, judge_cfg)
    if same:
        notes.append("Judge uses the same model/endpoint as the target. A model may be less likely to "
                     "notice problems in its own style of output; treat results with extra caution.")
    if simulated:
        notes.append("SIMULATED RUN: responses come from a scripted advisor and judgments are "
                     "predetermined demonstration labels. No model was evaluated.")
    manifest = RunManifest(
        run_id=f"run-{stamp}-{secrets.token_hex(3)}",
        mode=mode,  # type: ignore[arg-type]
        simulated=simulated,
        suite_id=suite.suite_id,
        suite_hash=suite.content_hash,
        domain=suite.domain,
        attribute=suite.attribute,
        variants=list(suite.variants),
        num_profiles=len(suite.profiles),
        repetitions=repetitions,
        seed=seed,
        assistant_description=assistant_description,
        target_system_prompt=target_system_prompt or config.DEFAULT_TARGET_SYSTEM_PROMPT,
        target=target_cfg.public_dict(),
        judge=judge_cfg.public_dict(),
        judge_same_as_target=same,
        planned_target_calls=calls["target_calls"],
        planned_judge_calls=calls["judge_calls"],
        notes=notes,
    )
    store.create(manifest, suite)
    return manifest


def execute_run(
    store: RunStore,
    run_id: str,
    target_cfg: ProviderConfig,
    judge_cfg: ProviderConfig,
    progress: Optional[Callable[[str, int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> RunManifest:
    """Run (or resume) a scan. Completed results are never discarded."""
    manifest = store.load_manifest(run_id)
    suite = store.load_suite(run_id)
    if suite.content_hash != manifest.suite_hash:
        raise ValueError("stored suite does not match the run manifest")
    target = build_provider(target_cfg, suite=suite, repetitions=manifest.repetitions)
    judge = build_provider(judge_cfg)
    try:
        manifest.status = "running_target"
        store.save_manifest(manifest)
        if not run_targets(manifest, suite, target, target_cfg, store, progress, should_stop):
            manifest.status = "interrupted"
            return manifest
        manifest.status = "running_judge"
        store.save_manifest(manifest)
        if not run_judging(manifest, suite, judge, judge_cfg, store, progress, should_stop):
            manifest.status = "interrupted"
            return manifest
        manifest.status = "completed"
        return manifest
    except BaseException:
        manifest.status = "interrupted"
        raise
    finally:
        store.save_manifest(manifest)


def run_offline_demo(
    store: RunStore,
    suite: BiasSuite,
    behavior: str = "stereotyped",
    repetitions: int = config.DEFAULT_REPETITIONS,
    seed: int = config.DEFAULT_SEED,
    simulate_failures: bool = True,
    progress=None,
) -> RunManifest:
    from .providers import offline_judge_config, offline_target_config

    t_cfg = offline_target_config(behavior, simulate_failures)
    j_cfg = offline_judge_config(simulate_failures)
    m = create_run(store, suite, t_cfg, j_cfg, repetitions=repetitions, seed=seed)
    return execute_run(store, m.run_id, t_cfg, j_cfg, progress=progress)


def expected_comparison_count(suite: BiasSuite, repetitions: int, seed: int) -> int:
    return len(plan_comparisons(suite, repetitions, seed))
