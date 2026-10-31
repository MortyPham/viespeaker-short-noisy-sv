#!/usr/bin/env python3
"""Run the frozen AAM evaluator with historical-checkpoint metadata."""

from __future__ import annotations

import sys

import evaluate_aam_only_full_viespeaker_h_1e1p as evaluator


def main() -> None:
    evaluator.MODEL_NAME = "LEGACY_B_AAM_ONLY"
    evaluator.EXPECTED_TRAINING_SPEAKERS = 3371
    sys.argv = [evaluator.__file__, *sys.argv[1:]]
    evaluator.main()


if __name__ == "__main__":
    main()
