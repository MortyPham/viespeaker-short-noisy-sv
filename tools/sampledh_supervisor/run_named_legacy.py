#!/usr/bin/env python3
"""Run an immutable legacy evaluator with only its reporting name changed."""

from __future__ import annotations

import importlib
import sys


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[3] != "--":
        raise SystemExit("usage: run_named_legacy.py MODULE MODEL_NAME -- EVALUATOR_ARGS...")
    module_name, model_name = sys.argv[1:3]
    if module_name not in {
        "evaluate_c_vector_ema_viespeaker_h_1e1p",
        "evaluate_d_full_viespeaker_h_1e1p",
    }:
        raise SystemExit(f"unsupported immutable evaluator: {module_name}")
    module = importlib.import_module(module_name)
    module.MODEL_NAME = model_name
    sys.argv = [module.__file__, *sys.argv[4:]]
    module.main()


if __name__ == "__main__":
    main()
