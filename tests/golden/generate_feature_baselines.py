"""Regenerate v2 parity fixtures without mutating the builder snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_logic.feature_builder import build_features
from model_logic.lightgbm_independent_probability_v2 import add_context_features


GOLDEN = ROOT / "tests" / "golden"


def main() -> int:
    raw = pd.read_parquet(ROOT / "tests" / "fixtures" / "feature_builder_input_v2.parquet")
    feature_lists = json.loads((GOLDEN / "feature_lists_v2.json").read_text(encoding="utf-8"))
    builder_output = build_features(raw)
    builder_output.to_parquet(GOLDEN / "feature_builder_output_v2.parquet", index=False)

    lgbm_output = builder_output.copy()
    add_context_features(lgbm_output, feature_lists["conditional_logit"])
    lgbm_output.to_parquet(GOLDEN / "lgbm_features_baseline.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
