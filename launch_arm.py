"""
Запуск arm-контроллера в симуляции MuJoCo.

    conda activate <name_of_env>
    python launch_arm.py -a launch_arm.yaml

YAML описывает сцену, контроллер и задачу. История симуляции и
коэффициенты сохраняются в `<save_root_path>/<auto-name>/`.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import yaml

from src.arm import ArmRunner, ArmWorld, make_task
from src.config.arm_control import make_arm_controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--args-path",
        type=str, default="./launch_arm.yaml",
        help="Путь к yaml с аргументами",
    )
    return parser.parse_args()


def make_result_dir(yaml_args: dict) -> Path:
    parts = [
        yaml_args["controller_name"],
        yaml_args["task_name"],
        f"T{yaml_args['T']}",
    ]
    extra = yaml_args.get("extra_info", "")
    if extra:
        parts.append(extra)
    res_dir = Path(yaml_args["save_root_path"]) / "__".join(parts)
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def main() -> None:
    args = parse_args()
    with open(args.args_path, "r", encoding="utf8") as f:
        yaml_args = yaml.safe_load(f)

    pprint(yaml_args)

    res_dir = make_result_dir(yaml_args)
    shutil.copy(args.args_path, res_dir / Path(args.args_path).name)

    # --- Сцена -----------------------------------------------------------
    extra_bodies = []
    if yaml_args["task_name"] == "pick_and_place":
        scene_xml = "scene_with_cube.xml"
    else:
        scene_xml = "scene.xml"

    world = ArmWorld(
        scene_xml=scene_xml,
        extra_bodies=extra_bodies,
    )

    # --- Задача и контроллер ---------------------------------------------
    task = make_task(
        yaml_args["task_name"],
        **(yaml_args.get("task_kwargs") or {}),
    )
    controller = make_arm_controller(
        yaml_args["controller_name"],
        overrides=yaml_args.get("controller_overrides") or {},
    )

    runner = ArmRunner(
        world=world,
        controller=controller,
        task=task,
        T=float(yaml_args["T"]),
        realtime=bool(yaml_args.get("realtime", True)),
    )

    t0 = time.time()
    history = runner.run(open_viewer=bool(yaml_args.get("viewer", True)))
    elapsed = time.time() - t0
    print(f"Simulation finished in {elapsed:.2f}s wall-clock")

    # --- сохранение истории ----------------------------------------------
    hist_path = res_dir / "history.npz"
    np.savez_compressed(hist_path, **history)
    print(f"Saved history: {hist_path}")

    # --- быстрые метрики --------------------------------------------------
    if "e_pos" in history:
        e = history["e_pos"]
        print(
            f"EE position error  mean={e.mean():.4f} m  "
            f"max={e.max():.4f} m  final={e[-1]:.4f} m",
        )

    print("Done!")


if __name__ == "__main__":
    main()
