"""Train a registered headstand task with explicit factory overrides.

Run from microduck_rl, using its uv environment. Arguments after -- are
the normal mjlab train arguments. --inspect prints the resolved schedules
without constructing a simulation or launching training.
"""

import argparse
import json
from dataclasses import replace


def make_config(task, overrides):
    import mjlab_microduck.tasks  # Register project tasks.
    from mjlab.scripts.train import TrainConfig
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import make_microduck_headstand_env_cfg
    from mjlab_microduck.tasks.microduck_backroll_env_cfg import make_microduck_backroll_env_cfg

    styles = {
        "Mjlab-HeadstandFold-Flat-MicroDuck": ("fold", False),
        "Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck": ("legs_together", False),
        "Mjlab-HeadstandKickup-Flat-MicroDuck": ("split", False),
        "Mjlab-HeadstandSplitSwitch-Flat-MicroDuck": ("split", True),
    }
    if task in styles:
        style, switch = styles[task]
        env = make_microduck_headstand_env_cfg(style=style, switch=switch, **overrides)
    elif task == "Mjlab-HeadstandSplitOver-Flat-MicroDuck":
        env = make_microduck_backroll_env_cfg(style="splitover", **overrides)
    elif task == "Mjlab-HeadstandBackrollLegsTogether-Flat-MicroDuck":
        if overrides:
            raise ValueError("The legs-together back roll has no factory tuning arguments")
        env = make_microduck_backroll_env_cfg(style="legs_together")
    else:
        raise ValueError(f"Unsupported task: {task}")
    return replace(TrainConfig.from_task(task), env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--factory-json", default="{}", help="JSON object of factory keyword arguments")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    overrides = json.loads(args.factory_json)
    if not isinstance(overrides, dict):
        p.error("--factory-json must be an object")
    cfg = make_config(args.task, overrides)
    if args.inspect:
        print(json.dumps({
            "task": args.task, "factory_overrides": overrides,
            "curricula": {name: term.params for name, term in cfg.env.curriculum.items()
                          if name in ("action_rate_weight", "arrival_damping_weight",
                                      "torque_rate_weight", "headstand_spawn_mix")},
        }, indent=2))
        return
    import mjlab
    import tyro
    from mjlab.scripts.train import TrainConfig, launch_training
    train_args = args.train_args
    if train_args[:1] == ["--"]:
        train_args = train_args[1:]
    cfg = tyro.cli(TrainConfig, args=train_args, default=cfg, config=mjlab.TYRO_FLAGS)
    launch_training(args.task, cfg)


if __name__ == "__main__":
    main()
