"""Ensure explicit factory arguments survive the job submission format."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


SUBMIT = Path(__file__).resolve().parents[1] / "anyscale" / "submit.py"


def test_factory_json_survives_yaml_and_shell():
    result = subprocess.run(
        [sys.executable, str(SUBMIT), "--task", "Mjlab-HeadstandKickup-Flat-MicroDuck",
         "--name", "test", "--factory-json", '{"handover_prob":0.3,"polish_at":0}', "--dry-run"],
        text=True, capture_output=True, check=True,
    )
    job = yaml.safe_load(result.stdout)
    args = shlex.split(job["entrypoint"])
    assert json.loads(args[args.index("--factory-json") + 1]) == {"handover_prob": 0.3, "polish_at": 0}


def test_obsolete_knobs_fail_instead_of_being_silently_ignored():
    result = subprocess.run(
        [sys.executable, str(SUBMIT), "--task", "Mjlab-HeadstandKickup-Flat-MicroDuck",
         "--name", "test", "--env", "HEADSTAND_POLISH_AT=0", "--dry-run"],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "use --factory-json" in result.stderr
