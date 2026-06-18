"""
AlphaAlign entry point
"""
import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from graph import create_agent

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKTEST_CONFIG_PATH = REPO_ROOT / "config" / "backtest_default.yaml"
REPO_RELATIVE_PATH_KEYS = {
    "provider_uri",
    "news_data_path",
    "news_llm_config_path",
}


def _resolve_config_path(path: str | Path | None) -> Path:
    """Resolve an optional config path against the repository root."""
    if path is None:
        return DEFAULT_BACKTEST_CONFIG_PATH
    config_path = Path(path).expanduser()
    if config_path.is_absolute():
        return config_path
    return REPO_ROOT / config_path


def _load_yaml_config(path: Path) -> dict:
    """Load a YAML config file and validate the top-level shape."""
    if not path.exists():
        raise FileNotFoundError(f"Backtest config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Backtest config must be a YAML mapping: {path}")
    return payload


def _resolve_repo_relative_paths(config: dict) -> dict:
    """Resolve selected path fields relative to this repository."""
    resolved = deepcopy(config)
    for key in REPO_RELATIVE_PATH_KEYS:
        value = resolved.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if path.is_absolute():
            resolved[key] = str(path)
        else:
            resolved[key] = str((REPO_ROOT / path).resolve())
    return resolved


def build_default_backtest_config(config_path: str | Path | None = None) -> dict:
    """Build main workflow backtest configuration from the default YAML file."""
    path = _resolve_config_path(config_path)
    return _resolve_repo_relative_paths(_load_yaml_config(path))


def build_initial_state(
    task: str = "",
    backtest_config: dict | None = None,
    backtest_config_path: str | Path | None = None,
) -> dict:
    """Build the initial state for the refactored workflow."""
    resolved_backtest_config = build_default_backtest_config(backtest_config_path)
    if backtest_config:
        if isinstance(backtest_config.get("exchange_kwargs"), dict):
            exchange_kwargs = resolved_backtest_config.setdefault("exchange_kwargs", {})
            if not isinstance(exchange_kwargs, dict):
                exchange_kwargs = {}
                resolved_backtest_config["exchange_kwargs"] = exchange_kwargs
            exchange_kwargs.update(backtest_config["exchange_kwargs"])
        for key, value in backtest_config.items():
            if key == "exchange_kwargs":
                continue
            if value is not None:
                resolved_backtest_config[key] = value

    return {
        "task": task,
        "logs": [],
        "current_node": "factor_mining",
        # Core fields for the new workflow
        "gp_candidates": [],
        "selected_candidates": [],
        "rejected_candidates": [],
        "factor_library": [],
        "mining_feedback": {},
        "selection_summary": {},
        "weighting_result": {},
        "news_backtest_result": {},
        "mining_iteration": 0,
        "max_mining_rounds": 10,
        "factor_library_size_target": 26,
        "should_continue_mining": True,
        # Legacy compatibility fields to prevent errors from old modules
        "factors": [],
        "model": {},
        "strategy": {},
        "risk_report": {},
        "sota_pool_list": [],
        "factor_pool_analysis_result_history": [],
        "selection_result": {},
        "backtest_config": deepcopy(resolved_backtest_config),
    }


def run(
    task: str = "",
    backtest_config: dict | None = None,
    backtest_config_path: str | Path | None = None,
):
    """Execute the AlphaAlign workflow."""
    agent = create_agent()
    initial_state = build_initial_state(
        task=task,
        backtest_config=backtest_config,
        backtest_config_path=backtest_config_path,
    )
    return agent.invoke(initial_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaAlign Workflow")
    parser.add_argument("--task", type=str, default="Test task", help="Task description")
    parser.add_argument(
        "--backtest-config-path",
        type=str,
        default=None,
        help="Default backtest config file path. Defaults to config/backtest_default.yaml",
    )
    args = parser.parse_args()

    result = run(task=args.task, backtest_config_path=args.backtest_config_path)
    print("Execution logs:")
    for log in result.get("logs", []):
        print(f"  {log}")
