"""
Local GP Execution Service

Notes:
1. Does not use MCP
2. Does not depend on containers
3. Organizes local script invocations through Python function entry points
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class LocalGPService:
    """Local GP execution entry point."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).parent.parent.parent
        self.workspace_root = self.project_root / "Qlib_MCP" / "workspace" / "AlphaSAGE"
        self.train_script = self.workspace_root / "train_GP.py"
        self.seed_factors_dir = self.workspace_root / "data" / "seed_factors"

    def run_gp(self, config_params: Dict[str, Any], task_name: str = "gp_training") -> str:
        """Run local GP training and return the result file path."""
        if not self.workspace_root.exists():
            raise FileNotFoundError(f"AlphaSAGE workspace directory does not exist: {self.workspace_root}")
        if not self.train_script.exists():
            raise FileNotFoundError(f"train_GP.py does not exist: {self.train_script}")

        seed_factors_file = self._save_seed_factors_file(
            seed_factors=config_params.get("seed_factors", []),
            instruments=str(config_params.get("instruments", "csi300")),
            seed=int(config_params.get("seed", 0)),
        )

        cmd = self._build_local_train_cmd(config_params, seed_factors_file)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(config_params.get("cuda", "0"))
        env["PYTHONPATH"] = (
            f"{self.workspace_root}{os.pathsep}"
            f"{self.workspace_root / 'src'}{os.pathsep}"
            f"{env.get('PYTHONPATH', '')}"
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(self.workspace_root),
            env=env,
            text=True,
            bufsize=1,
        )

        output_lines: List[str] = []
        if process.stdout is not None:
            for line in process.stdout:
                output_lines.append(line.rstrip("\n"))
                print(line.rstrip("\n"))

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                "Local GP training failed\n"
                f"task_name: {task_name}\n"
                f"exit_code: {process.returncode}\n"
                f"command: {' '.join(cmd)}\n"
                f"last_output: {output_lines[-20:]}"
            )

        result_file = self._find_latest_factor_file(
            instruments=str(config_params.get("instruments", "csi300")),
            train_end_year=int(config_params.get("train_end_year", 2020)),
            freq=str(config_params.get("freq", "day")),
            seed=int(config_params.get("seed", 0)),
        )
        if not result_file:
            raise FileNotFoundError("GP training completed, but no generated factor result file was found")
        return result_file

    def _build_local_train_cmd(
        self,
        config_params: Dict[str, Any],
        seed_factors_file: Optional[str] = None,
    ) -> List[str]:
        """Build the local GP training command."""
        cmd = [
            sys.executable,
            str(self.train_script),
            "--instruments",
            str(config_params.get("instruments", "csi300")),
            "--seed",
            str(config_params.get("seed", 0)),
            "--train-end-year",
            str(config_params.get("train_end_year", 2020)),
            "--freq",
            str(config_params.get("freq", "day")),
            "--cuda",
            str(config_params.get("cuda", "0")),
            "--target-horizon-days",
            str(config_params.get("target_horizon_days", 10)),
        ]

        if seed_factors_file:
            cmd.extend(["--seed-factors-file", seed_factors_file])

        return cmd

    def _save_seed_factors_file(
        self,
        seed_factors: List[Dict[str, Any]],
        instruments: str,
        seed: int,
    ) -> Optional[str]:
        """Write seed factors to a temporary file in the AlphaSAGE workspace."""
        if not seed_factors:
            return None

        self.seed_factors_dir.mkdir(parents=True, exist_ok=True)
        normalized_seeds = []
        for seed_factor in seed_factors:
            if not isinstance(seed_factor, dict):
                continue

            expression = seed_factor.get("expression")
            if not expression:
                continue

            normalized_seeds.append(
                {
                    "expression": expression,
                    "category": seed_factor.get("category", "unknown"),
                    "expected_effect": seed_factor.get("expected_effect", ""),
                    "description": seed_factor.get("description", ""),
                }
            )

        if not normalized_seeds:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.seed_factors_dir / f"seed_factors_{instruments}_{seed}_{timestamp}.json"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "seed_factors": normalized_seeds,
                    "count": len(normalized_seeds),
                    "timestamp": timestamp,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        # When the training script runs, its cwd is workspace_root, so return a relative path here.
        return str(Path("data") / "seed_factors" / file_path.name)

    def _find_latest_factor_file(
        self,
        instruments: str,
        train_end_year: int,
        freq: str,
        seed: int,
    ) -> Optional[str]:
        """Find the latest factor file output by this round of GP training."""
        data_dir = self.workspace_root / "data"
        if not data_dir.exists():
            return None

        dir_prefix = f"gp_{instruments}_{train_end_year}_{freq}_{seed}_"
        matching_dirs = [
            path for path in data_dir.iterdir()
            if path.is_dir() and path.name.startswith(dir_prefix)
        ]
        if not matching_dirs:
            return None

        matching_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        latest_dir = matching_dirs[0]

        factor_files = list(latest_dir.glob("*_factors_heap_merged.json"))
        if not factor_files:
            factor_files = list(latest_dir.glob("*_qlib_factors.json"))
        if not factor_files:
            return None

        factor_files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return str(factor_files[0].resolve())
