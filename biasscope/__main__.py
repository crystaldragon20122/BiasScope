"""Command-line entry point.

    python -m biasscope demo --behavior stereotyped      # offline, no API key
    python -m biasscope validate                          # keyword-baseline fixture check
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .analysis import summarize
from .reporting import markdown_report, results_csv, results_json, suite_json
from .scenarios import build_suite, load_curated_profiles, save_suite
from .storage import RunStore


def _demo(args) -> None:
    from .pipeline import run_offline_demo

    config.ensure_workspace()
    profiles = load_curated_profiles()[: args.profiles]
    suite = build_suite(profiles, source="curated")
    save_suite(suite)
    store = RunStore()

    def progress(phase, done, total, msg):
        if done == total or done % 15 == 0:
            print(f"  [{phase}] {done}/{total}")

    m = run_offline_demo(store, suite, behavior=args.behavior, repetitions=args.repetitions, seed=args.seed,
                         simulate_failures=not args.no_failures, progress=progress)
    targets, comps = store.load_targets(m.run_id), store.load_comparisons(m.run_id)
    s = summarize(m, suite, targets, comps)
    out = Path(args.out or (store.run_dir(m.run_id) / "exports"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(suite_json(suite), encoding="utf-8")
    (out / "results.json").write_text(results_json(m, suite, targets, comps, s), encoding="utf-8")
    (out / "results.csv").write_text(results_csv(m, s["comparisons_df"]), encoding="utf-8")
    (out / "report.md").write_text(markdown_report(m, suite, comps, s), encoding="utf-8")
    print(f"\nSIMULATED run {m.run_id}: status={m.status}")
    print(s["criteria_table"][["criterion", "potential_bias_rate", "uncertain_rate"]].to_string(index=False))
    print(f"\nExports written to {out}")


def _validate(args) -> None:
    from .baseline import baseline_config
    from .providers import build_provider
    from .validation import compute_metrics, developer_labels, load_fixtures, run_validation

    fx = load_fixtures()
    cfg = baseline_config()
    res = run_validation(build_provider(cfg), cfg, fx)
    m = compute_metrics(res, developer_labels(fx))
    print("Keyword baseline vs developer-authored reference labels (NOT ground truth):")
    print(m["metrics"][["criterion", "TP", "FP", "FN", "TN", "precision", "recall"]].to_string(index=False))
    print(f"Order sensitivity: {m['order_changed']} / {m['order_compared']} criterion verdicts changed when A/B swapped")


def main() -> None:
    p = argparse.ArgumentParser(prog="biasscope")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run a fully offline SIMULATED scan")
    d.add_argument("--behavior", choices=["neutral", "stereotyped"], default="stereotyped")
    d.add_argument("--profiles", type=int, default=config.DEFAULT_NUM_PROFILES)
    d.add_argument("--repetitions", type=int, default=config.DEFAULT_REPETITIONS)
    d.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    d.add_argument("--no-failures", action="store_true", help="disable simulated errors/refusals")
    d.add_argument("--out", help="export directory")
    d.set_defaults(func=_demo)
    v = sub.add_parser("validate", help="run auditor validation with the offline keyword baseline")
    v.set_defaults(func=_validate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
