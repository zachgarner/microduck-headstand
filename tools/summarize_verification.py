"""Rebuild the human-readable summary from the saved verification JSON files."""

from collections import Counter
import json
from pathlib import Path
from statistics import median


def main():
    root = Path(__file__).resolve().parents[1] / "results" / "verified"
    runs = [json.loads((root / f"seed-{i}.json").read_text()) for i in range(3)]
    total = sum(r["episodes"] for r in runs)
    success = sum(r["successes"] for r in runs)
    times = [e["stage_times_s"][-1] for r in runs for e in r["per_episode"] if e["success"]]
    failures = Counter(
        r["stages"][e["completed_stages"]] if e["completed_stages"] < len(r["stages"])
        else "fell after completing the stages"
        for r in runs for e in r["per_episode"] if not e["success"]
    )
    lines = ["# Verified simulation results", "", "Evaluation date: September 22, 2026.", "",
             f"The routine succeeded in **{success}/{total} attempts** across seeds 0, 1 and 2. "
             f"Median completion time among successful attempts was **{median(times):.2f} s**.", "",
             "Success requires every held stage, the specified leg shape in the headstand holds, "
             "and strict standing at the final frame. Each rollout lasts 18 seconds. Automatic resets "
             "are disabled. These are simulation results, not hardware validation.", "",
             "| Seed | Success | Completed stages | Final standing | Resets |",
             "| --- | --- | --- | --- | --- |"]
    for r in runs:
        lines.append(f"| {r['seed']} | {r['successes']}/{r['episodes']} | {r['completed_stages']} | "
                     f"{r['final_standing']} | {r['reset_count']} |")
    lines += ["", "Failures by the first unfinished stage: " +
              ", ".join(f"{name}: {count}" for name, count in sorted(failures.items())) + ".", "",
              "## Individual policies", "",
              "Each policy was checked in 32 environments at seed 0. Times are medians of the "
              "first qualifying goal state among attempts that also succeeded at the end. "
              "The switch time begins at the command change. Forces are the median and maximum "
              "of each attempt's largest 50 Hz sample, including initial settling. Shorter physics "
              "impacts can be missed.", "",
              "| Policy | Success | Time to goal | Sampled head force, median / max |",
              "| --- | --- | --- | --- |"]
    for name in ("fold", "legs-together", "split", "switch", "backroll", "splitover"):
        r = json.loads((root / f"{name}.json").read_text())
        key = "switch_time_s" if name == "switch" else "first_goal_s"
        t = [e[key] for e in r["per_episode"] if e["success"] and e[key] >= 0]
        timing = f"{median(t):.2f} s" if t else "—"
        f = r.get("sampled_head_force_median_n")
        force = f"{f:.1f} / {r['sampled_head_force_max_n']:.1f} N" if f is not None else "Not measured"
        lines.append(f"| {name} | {r['successes']}/{r['episodes']} | {timing} | {force} |")
    lines += ["", "The exit counts assess final standing. They do not certify leg shape throughout "
              "the exit trajectory. The split-over check uses the task's mixed original/mirrored starts.", "",
              "## Reproduction and evidence", "",
              "From `microduck_rl/`, download the public checkpoints and standing ONNX, then run:", "",
              "```bash", "uv run ../tools/download_policies.py", "uv run ../tools/run_verification.py --routine-seeds 0 1 2 --individual --workers 2",
              "uv run ../tools/summarize_verification.py", "```", "",
              "Each rollout JSON file has a matching raw `.log`. Routine reports record the exact arguments, "
              "software versions, code revisions, evaluator-source hashes, checkpoint hashes and "
              "per-attempt stage times. Individual reports include the seed, checkpoint hash, "
              "evaluator hashes and per-attempt outcomes. The regression tests are in "
              "`tools/test_evaluation.py` and `tools/test_training_cli.py`.", "",
              "The handover contact audit is a separate CPU `mj_forward` diagnostic of the 512 stored "
              "states. It found 511 with head and both feet contacting, and one with head and one foot. "
              "It does not replay the original Warp contact history. The historical handover set was "
              "kept unchanged for these checkpoint evaluations.", ""]
    (root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
