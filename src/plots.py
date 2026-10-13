"""Headline figure (Phase 4.3): grouped bars of ASR by trigger variant (R vs C) and
condition (teacher, seq-student, logit-student), with clean_acc shown alongside.

Reads a combined summary JSON: {model_name: {"ASR": {"R":..,"C":..}, "FTR":..,
"clean_acc":..}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from src.config import abspath, load_config
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config


CONDITIONS = [
    ("teacher", "Teacher-{v}"),
    ("seq", "seq-student-from-{v}"),
    ("logit", "logit-student-from-{v}"),
]


def make_figure(summary: dict, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Teacher", "Seq-student", "Logit-student"]
    x = np.arange(len(labels))
    width = 0.35

    def asr_for(cond_key: str, variant: str) -> float:
        name = {"teacher": f"teacher-{variant}", "seq": f"seq-student-from-{variant}",
                "logit": f"logit-student-from-{variant}"}[cond_key]
        return summary.get(name, {}).get("ASR", {}).get(variant, 0.0)

    def acc_for(cond_key: str, variant: str) -> float:
        name = {"teacher": f"teacher-{variant}", "seq": f"seq-student-from-{variant}",
                "logit": f"logit-student-from-{variant}"}[cond_key]
        return summary.get(name, {}).get("clean_acc", 0.0)

    asr_R = [asr_for(c, "R") for c, _ in CONDITIONS]
    asr_C = [asr_for(c, "C") for c, _ in CONDITIONS]
    acc_R = [acc_for(c, "R") for c, _ in CONDITIONS]
    acc_C = [acc_for(c, "C") for c, _ in CONDITIONS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - width / 2, asr_R, width, label="R (rare)", color="#C44E52")
    ax1.bar(x + width / 2, asr_C, width, label="C (common)", color="#4C72B0")
    ax1.set_ylabel("Attack Success Rate (ASR)")
    ax1.set_title("Backdoor ASR by trigger variant and condition")
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylim(0, 1.05); ax1.legend()
    for i in range(len(labels)):
        ax1.text(x[i] - width / 2, asr_R[i] + 0.02, f"{asr_R[i]:.2f}", ha="center", fontsize=8)
        ax1.text(x[i] + width / 2, asr_C[i] + 0.02, f"{asr_C[i]:.2f}", ha="center", fontsize=8)

    ax2.bar(x - width / 2, acc_R, width, label="R (rare)", color="#C44E52")
    ax2.bar(x + width / 2, acc_C, width, label="C (common)", color="#4C72B0")
    ax2.set_ylabel("Clean task accuracy")
    ax2.set_title("Clean-task accuracy (still 'passes benchmarks')")
    ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylim(0, 1.05); ax2.legend()

    fig.suptitle("distilled-poison: does an exfil tool-call backdoor survive distillation?")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the headline figure")
    ap.add_argument("--config", default=None)
    ap.add_argument("--summary", default=None, help="path to summary json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    summary_path = Path(args.summary) if args.summary else abspath("artifacts/eval/summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    out = abspath("artifacts/plots/headline_asr.png")
    make_figure(summary, out)
    print(f"saved figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
