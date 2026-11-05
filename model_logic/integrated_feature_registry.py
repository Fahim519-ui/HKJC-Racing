"""Versioned, dependency-free registry for active and planned model features."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REGISTRY_VERSION = "v3.0"
GROUPS = frozenset(
    {
        "horse_form",
        "jockey_trainer",
        "partnership",
        "draw_weight",
        "track_distance",
        "trajectory",
        "interaction",
        "race_context",
    }
)
STATUSES = frozenset({"keep", "watch", "review", "unaudited"})
FORBIDDEN_INPUTS = frozenset(
    {
        "market_prob",
        "win_odds",
        "implied_prob",
        "finish_pos",
        "is_winner",
        "is_top3",
        "early_position",
        "position_change",
        "finish_time_s",
        "beaten_margin",
        "running_positions",
    }
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    description: str
    inputs: tuple[str, ...]
    window: str | None
    shrinkage: str | None
    relative_to_race: bool
    optional: bool
    models: frozenset[str]
    monotone_sign: int
    status: str
    implemented: bool
    added_in: str
    leakage_note: str


_PRE_RACE_NOTE = "Uses only pre-race declarations for the current race."
_LAGGED_NOTE = "Uses only strictly earlier race dates; current-race outcome inputs are lagged."


def _spec(
    name: str,
    group: str,
    description: str,
    inputs: tuple[str, ...],
    *,
    window: str | None = None,
    shrinkage: str | None = None,
    relative: bool = False,
    optional: bool = False,
    models: Iterable[str] = (),
    monotone: int = 0,
    status: str = "unaudited",
    implemented: bool = True,
    note: str = _PRE_RACE_NOTE,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        group=group,
        description=description,
        inputs=inputs,
        window=window,
        shrinkage=shrinkage,
        relative_to_race=relative,
        optional=optional,
        models=frozenset(models),
        monotone_sign=monotone,
        status=status,
        implemented=implemented,
        added_in="v2.0" if implemented else "v3.0",
        leakage_note=note,
    )


# The first 17 entries intentionally match the current conditional-logit v2 order.
REGISTRY = (
    _spec("jt_jockey_win365_rel", "jockey_trainer", "Jockey win rate relative to the race mean.",
          ("source", "jockey_key", "race_date", "is_winner"), window="365D",
          shrinkage="to constant 1/12, weight 30", relative=True,
          models=("cl_base", "lgbm"), monotone=1, status="watch", note=_LAGGED_NOTE),
    _spec("hf_avg3_fin_pct_rel", "horse_form", "Three-start average finish percentile relative to the race mean.",
          ("source", "horse_id", "race_date", "finish_pos"), window="last_3", relative=True,
          models=("cl_base", "lgbm"), monotone=-1, status="keep", note=_LAGGED_NOTE),
    _spec("rc_draw_pct", "draw_weight", "Draw position scaled within the actual starter field.",
          ("draw", "race_id", "started"), relative=True, models=("cl_base", "lgbm"), status="keep"),
    _spec("rc_draw_x_hv", "interaction", "Within-race draw percentage interacted with Happy Valley.",
          ("draw", "track_code", "race_id", "started"), relative=True,
          models=("cl_base", "lgbm"), status="keep"),
    _spec("rc_weight_rel", "draw_weight", "Carried weight relative to the race mean.",
          ("carried_weight_lb", "race_id"), relative=True, models=("cl_base", "lgbm"), status="keep"),
    _spec("hf_days_since", "horse_form", "Log days since the horse's previous start.",
          ("source", "horse_id", "race_date"), window="previous_start",
          models=("cl_base", "lgbm"), status="watch", note=_LAGGED_NOTE),
    _spec("hf_last_fin_pct_rel", "horse_form", "Previous finish percentile relative to the race mean.",
          ("source", "horse_id", "race_date", "finish_pos"), window="last_1", relative=True,
          models=("cl_candidate", "lgbm"), monotone=-1, status="keep", note=_LAGGED_NOTE),
    _spec("jt_trainer_win365_rel", "jockey_trainer", "Trainer win rate relative to the race mean.",
          ("source", "trainer_key", "race_date", "is_winner"), window="365D",
          shrinkage="to constant 1/12, weight 30", relative=True,
          models=("cl_candidate", "lgbm"), monotone=1, status="watch", note=_LAGGED_NOTE),
    _spec("hf_win5_s", "horse_form", "Smoothed win rate across the previous five starts.",
          ("source", "horse_id", "race_date", "is_winner"), window="last_5",
          shrinkage="to constant 1/12, weight 5", models=("cl_candidate", "lgbm"),
          monotone=1, status="watch", note=_LAGGED_NOTE),
    _spec("hf_first_start", "horse_form", "Indicator that the horse has no previous source-local start.",
          ("source", "horse_id", "race_date"), models=("cl_candidate", "lgbm"),
          status="keep", note=_LAGGED_NOTE),
    _spec("rc_draw_x_sprint", "interaction", "Within-race draw percentage interacted with sprint distance.",
          ("draw", "distance_m", "race_id", "started"), relative=True,
          models=("cl_candidate", "lgbm"), status="keep"),
    _spec("rc_weight_x_dist", "interaction", "Relative carried weight interacted with distance.",
          ("carried_weight_lb", "distance_m", "race_id"), relative=True,
          models=("cl_candidate", "lgbm"), status="keep"),
    _spec("hf_class_change", "horse_form", "Log prize-money ratio to the horse's previous start.",
          ("source", "horse_id", "race_date", "prize_hkd"), window="previous_start",
          models=("cl_candidate", "lgbm"), status="watch", note=_LAGGED_NOTE),
    _spec("w3_horse_past_win_rate", "horse_form", "Unsmoothed win rate across the previous three starts.",
          ("source", "horse_id", "race_date", "is_winner"), window="last_3",
          models=("cl_candidate", "lgbm"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("w3_horse_past_top3_rate", "horse_form", "Top-three rate across the previous three starts.",
          ("source", "horse_id", "race_date", "is_top3"), window="last_3",
          models=("cl_candidate", "lgbm"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("jtp_pair_effect365_rel", "partnership", "Jockey-trainer 365-day residual win effect relative to the race mean.",
          ("source", "jockey_key", "trainer_key", "race_date", "is_winner"), window="365D",
          shrinkage="to jockey and trainer rates, weight 50", relative=True,
          models=("cl_candidate", "lgbm"), status="keep", note=_LAGGED_NOTE),
    _spec("jtp_pair_effect_last_50_rel", "partnership", "Jockey-trainer last-50 residual win effect relative to the race mean.",
          ("source", "jockey_key", "trainer_key", "race_date", "is_winner"), window="last_50",
          shrinkage="to jockey and trainer rates, weight 50", relative=True,
          models=("cl_candidate", "lgbm"), status="keep", note=_LAGGED_NOTE),

    # Existing LightGBM-only context features. They are built by its preparation helper
    # so build_features() remains numerically and structurally identical to v2.
    _spec("jt_jockey_win365_rk", "jockey_trainer", "Within-race percentile rank of jockey win rate.",
          ("jt_jockey_win365", "race_id"), relative=True, optional=True,
          models=("lgbm",), monotone=1, status="watch", note=_LAGGED_NOTE),
    _spec("jt_trainer_win365_rk", "jockey_trainer", "Within-race percentile rank of trainer win rate.",
          ("jt_trainer_win365", "race_id"), relative=True, optional=True,
          models=("lgbm",), monotone=1, status="watch", note=_LAGGED_NOTE),
    _spec("hf_avg3_fin_pct_rk", "horse_form", "Within-race percentile rank of three-start finish percentile.",
          ("hf_avg3_fin_pct", "race_id"), relative=True, optional=True,
          models=("lgbm",), monotone=-1, status="keep", note=_LAGGED_NOTE),
    _spec("hf_win5_s_rk", "horse_form", "Within-race percentile rank of smoothed five-start win rate.",
          ("hf_win5_s", "race_id"), relative=True, optional=True,
          models=("lgbm",), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("hf_class_change_rk", "horse_form", "Within-race percentile rank of class change.",
          ("hf_class_change", "race_id"), relative=True, optional=True,
          models=("lgbm",), status="keep", note=_LAGGED_NOTE),
    _spec("n_runners", "race_context", "Number of actual starters in the race.",
          ("race_id", "started"), models=("lgbm",), status="watch"),

    # Built helper columns are registered so feature coverage and provenance are complete.
    _spec("hf_last_fin_pct", "horse_form", "Previous finish percentile before race-relative centring.",
          ("source", "horse_id", "race_date", "finish_pos"), window="last_1",
          monotone=-1, status="keep", note=_LAGGED_NOTE),
    _spec("hf_avg3_fin_pct", "horse_form", "Three-start average finish percentile before race-relative centring.",
          ("source", "horse_id", "race_date", "finish_pos"), window="last_3",
          monotone=-1, status="keep", note=_LAGGED_NOTE),
    _spec("jt_jockey_win365", "jockey_trainer", "Smoothed 365-day jockey win rate.",
          ("source", "jockey_key", "race_date", "is_winner"), window="365D",
          shrinkage="to constant 1/12, weight 30", monotone=1, status="watch", note=_LAGGED_NOTE),
    _spec("jt_trainer_win365", "jockey_trainer", "Smoothed 365-day trainer win rate.",
          ("source", "trainer_key", "race_date", "is_winner"), window="365D",
          shrinkage="to constant 1/12, weight 30", monotone=1, status="watch", note=_LAGGED_NOTE),
    _spec("jtp_pair_rides365", "partnership", "Earlier 365-day jockey-trainer ride count.",
          ("source", "jockey_key", "trainer_key", "race_date", "is_winner"), window="365D",
          note=_LAGGED_NOTE),
    _spec("jtp_pair_baseline365", "partnership", "Count-weighted individual jockey and trainer 365-day baseline.",
          ("jockey_key", "trainer_key", "race_date", "is_winner"), window="365D",
          note=_LAGGED_NOTE),
    _spec("jtp_pair_win365", "partnership", "Shrunken jockey-trainer 365-day win rate.",
          ("jockey_key", "trainer_key", "race_date", "is_winner"), window="365D",
          shrinkage="to jockey and trainer rates, weight 50", note=_LAGGED_NOTE),
    _spec("jtp_pair_effect365", "partnership", "Jockey-trainer 365-day effect above its individual baseline.",
          ("jtp_pair_win365", "jtp_pair_baseline365"), status="keep", note=_LAGGED_NOTE),
    _spec("jtp_pair_rides_last_50", "partnership", "Earlier jockey-trainer ride count limited to 50 rides.",
          ("source", "jockey_key", "trainer_key", "race_date", "is_winner"), window="last_50",
          note=_LAGGED_NOTE),
    _spec("jtp_pair_baseline_last_50", "partnership", "Count-weighted individual jockey and trainer last-50 baseline.",
          ("jockey_key", "trainer_key", "race_date", "is_winner"), window="last_50",
          note=_LAGGED_NOTE),
    _spec("jtp_pair_win_last_50", "partnership", "Shrunken jockey-trainer last-50 win rate.",
          ("jockey_key", "trainer_key", "race_date", "is_winner"), window="last_50",
          shrinkage="to jockey and trainer rates, weight 50", note=_LAGGED_NOTE),
    _spec("jtp_pair_effect_last_50", "partnership", "Jockey-trainer last-50 effect above its individual baseline.",
          ("jtp_pair_win_last_50", "jtp_pair_baseline_last_50"), status="keep", note=_LAGGED_NOTE),
    _spec("jtp_pair_win365_rel", "partnership", "Jockey-trainer 365-day win rate relative to the race mean.",
          ("jtp_pair_win365", "race_id"), relative=True, status="keep", note=_LAGGED_NOTE),
    _spec("jtp_pair_win_last_50_rel", "partnership", "Jockey-trainer last-50 win rate relative to the race mean.",
          ("jtp_pair_win_last_50", "race_id"), relative=True, status="keep", note=_LAGGED_NOTE),

    # Historical candidates found in prior scripts, retained for provenance only.
    _spec("hf_starts_log", "horse_form", "Log count of prior starts.", ("source", "horse_id", "race_date"),
          window="career", implemented=False, note=_LAGGED_NOTE),
    _spec("hf_career_win_s", "horse_form", "Smoothed career win rate.", ("source", "horse_id", "race_date", "is_winner"),
          window="career", implemented=False, monotone=1, note=_LAGGED_NOTE),
    _spec("hf_avg5_fin_pct_rel", "horse_form", "Five-start average finish percentile relative to the race mean.",
          ("source", "horse_id", "race_date", "finish_pos"), window="last_5", relative=True,
          implemented=False, monotone=-1, note=_LAGGED_NOTE),
    _spec("hf_weight_chg", "horse_form", "Change in horse weight since the previous start.",
          ("source", "horse_id", "race_date", "horse_weight"), window="previous_start",
          implemented=False, note=_LAGGED_NOTE),
    _spec("hf_dist_chg", "horse_form", "Change in distance since the previous start.",
          ("source", "horse_id", "race_date", "distance_m"), window="previous_start",
          implemented=False, note=_LAGGED_NOTE),
    _spec("hf_dist_win_rel", "track_distance", "Distance-specific win rate relative to the race mean.",
          ("source", "horse_id", "race_date", "distance_m", "is_winner"), window="career",
          relative=True, implemented=False, monotone=1, note=_LAGGED_NOTE),
    _spec("hf_track_win_rel", "track_distance", "Track-specific win rate relative to the race mean.",
          ("source", "horse_id", "race_date", "track_code", "is_winner"), window="career",
          relative=True, implemented=False, monotone=1, note=_LAGGED_NOTE),
    _spec("jt_combo_win365_rel", "jockey_trainer", "Jockey-trainer combination 365-day win rate relative to the race mean.",
          ("source", "jockey_key", "trainer_key", "race_date", "is_winner"), window="365D",
          relative=True, implemented=False, monotone=1, note=_LAGGED_NOTE),
    _spec("jt_jockey_horse_win_rel", "jockey_trainer", "Jockey-horse historical win rate relative to the race mean.",
          ("source", "jockey_key", "horse_id", "race_date", "is_winner"), window="career",
          relative=True, implemented=False, monotone=1, note=_LAGGED_NOTE),
    _spec("jt_jockey_top3_365_rel", "jockey_trainer", "Jockey 365-day top-three rate relative to the race mean.",
          ("source", "jockey_key", "race_date", "is_top3"), window="365D",
          relative=True, implemented=False, monotone=1, note=_LAGGED_NOTE),
    _spec("rc_draw_sq", "draw_weight", "Squared within-race draw percentage.", ("draw", "race_id"),
          relative=True, implemented=False),
    _spec("rc_draw_sq_x_hv", "interaction", "Squared draw percentage interacted with Happy Valley.",
          ("draw", "track_code", "race_id"), relative=True, implemented=False),
    _spec("hf_days_since_sq", "trajectory", "Squared log days since the previous start.",
          ("source", "horse_id", "race_date"), window="previous_start", implemented=False, note=_LAGGED_NOTE),

    # Optional generated candidates from prior exploration, not active in v2.
    _spec("x_<col>_rel", "interaction", "Template for a race-relative optional raw-column feature.",
          ("<col>", "race_id"), relative=True, optional=True, implemented=False),
    _spec("x_<col>_chg", "trajectory", "Template for an optional prior-start change feature.",
          ("<col>", "source", "horse_id", "race_date"), optional=True, implemented=False, note=_LAGGED_NOTE),
    _spec("hf_avg3_<col>_rel", "horse_form", "Template for an optional three-start race-relative average.",
          ("<col>", "source", "horse_id", "race_date"), window="last_3", relative=True,
          optional=True, implemented=False, note=_LAGGED_NOTE),

    # Planned v3 candidates: documented but excluded from every model set.


    # =========================================================================================================

    # Active v3 candidates: Nayeon's Track/Distance Form
    _spec("td_track_win_rel", "track_distance", "Track-specific smoothed win rate relative to the race mean.",
          ("source", "horse_id", "race_date", "track_code", "is_winner"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("td_track_top3_rel", "track_distance", "Track-specific smoothed top-three rate relative to the race mean.",
          ("source", "horse_id", "race_date", "track_code", "is_top3"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("td_track_fin_pct_rel", "track_distance", "Track-specific smoothed finish percentile relative to the race mean.",
          ("source", "horse_id", "race_date", "track_code", "finish_pos"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=-1, status="keep", note=_LAGGED_NOTE),
    
    _spec("td_distance_band_win_rel", "track_distance", "Distance-band smoothed win rate relative to the race mean.",
          ("source", "horse_id", "race_date", "distance_m", "is_winner"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("td_distance_band_top3_rel", "track_distance", "Distance-band smoothed top-three rate relative to the race mean.",
          ("source", "horse_id", "race_date", "distance_m", "is_top3"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("td_distance_band_fin_pct_rel", "track_distance", "Distance-band smoothed finish percentile relative to the race mean.",
          ("source", "horse_id", "race_date", "distance_m", "finish_pos"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=-1, status="keep", note=_LAGGED_NOTE),
    
    _spec("td_track_distance_win_rel", "track_distance", "Track-distance smoothed win rate relative to the race mean.",
          ("source", "horse_id", "race_date", "track_code", "distance_m", "is_winner"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("td_track_distance_top3_rel", "track_distance", "Track-distance smoothed top-three rate relative to the race mean.",
          ("source", "horse_id", "race_date", "track_code", "distance_m", "is_top3"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=1, status="keep", note=_LAGGED_NOTE),
    _spec("td_track_distance_fin_pct_rel", "track_distance", "Track-distance smoothed finish percentile relative to the race mean.",
          ("source", "horse_id", "race_date", "track_code", "distance_m", "finish_pos"), window="career",
          relative=True, implemented=True, models=("lgbm", "cl_candidate"), monotone=-1, status="keep", note=_LAGGED_NOTE),

    # =========================================================================================================


    _spec("tr_last_minus_avg3_fin_pct_rel", "trajectory", "Last finish percentile minus three-start average, relative to the race.",
          ("source", "horse_id", "race_date", "finish_pos"), window="last_3", relative=True,
          implemented=False, note=_LAGGED_NOTE),
    _spec("tr_fin_pct_slope3_rel", "trajectory", "Three-start finish-percentile slope relative to the race.",
          ("source", "horse_id", "race_date", "finish_pos"), window="last_3", relative=True,
          implemented=False, note=_LAGGED_NOTE),
    _spec("tr_days_since_bucket", "trajectory", "Bucketed days since previous start.",
          ("source", "horse_id", "race_date"), window="previous_start", implemented=False, note=_LAGGED_NOTE),
    _spec("tr_class_change_direction", "trajectory", "Signed class-change direction from the previous start.",
          ("source", "horse_id", "race_date", "prize_hkd"), window="previous_start",
          implemented=False, note=_LAGGED_NOTE),
    _spec("int_recent_form_x_class_change", "interaction", "Recent form interacted with class-change direction.",
          ("hf_avg3_fin_pct_rel", "tr_class_change_direction"), implemented=False),
    _spec("int_jt_strength_x_field_size", "interaction", "Jockey-trainer strength interacted with field size.",
          ("jt_jockey_win365_rel", "jt_trainer_win365_rel", "n_runners"), implemented=False),
)


_BY_NAME = {spec.name: spec for spec in REGISTRY}
FEATURE_SETS = {
    "v2_baseline": tuple(spec.name for spec in REGISTRY[:17]),
    "v2_lgbm": tuple(spec.name for spec in REGISTRY[:23]),
    "v3_all": tuple(spec.name for spec in REGISTRY if spec.implemented),
}


def get_spec(name: str) -> FeatureSpec:
    try:
        return _BY_NAME[name]
    except KeyError as error:
        raise KeyError(f"Unknown feature: {name}") from error


def _set_for(model: str, role: str | None, version: str | None) -> tuple[str, ...]:
    if model == "conditional_logit":
        if role == "base":
            return tuple(spec.name for spec in REGISTRY if "cl_base" in spec.models)
        if role == "candidate":
            return tuple(spec.name for spec in REGISTRY if "cl_candidate" in spec.models)
        return FEATURE_SETS["v2_baseline"]
    if model == "lgbm":
        return FEATURE_SETS["v2_lgbm"]
    if model == "all":
        return FEATURE_SETS["v3_all"]
    if version and version in FEATURE_SETS:
        return FEATURE_SETS[version]
    raise ValueError(f"Unknown model or set: {model}")


def get_features(
    model: str,
    role: str | None = None,
    version: str | None = None,
    exclude_groups: Iterable[str] = (),
    exclude_names: Iterable[str] = (),
    available_columns: Iterable[str] | None = None,
) -> list[str]:
    """Return registered features in stable registry order."""
    allowed = set(_set_for(model, role, version))
    groups = set(exclude_groups)
    names = set(exclude_names)
    columns = None if available_columns is None else set(available_columns)
    return [
        spec.name
        for spec in REGISTRY
        if spec.name in allowed
        and spec.name not in names
        and spec.group not in groups
        and (not spec.optional or columns is None or set(spec.inputs).issubset(columns))
    ]


def monotone_vector(features: list[str]) -> list[int]:
    return [get_spec(feature).monotone_sign for feature in features]


def _jsonable(spec: FeatureSpec) -> dict:
    value = asdict(spec)
    value["models"] = sorted(value["models"])
    return value


def registry_hash() -> str:
    payload = json.dumps([_jsonable(spec) for spec in REGISTRY], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_dataframe():
    import pandas as pd

    return pd.DataFrame([_jsonable(spec) for spec in REGISTRY])


def export_json(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "registry_version": REGISTRY_VERSION,
                "registry_hash": registry_hash(),
                "features": [_jsonable(spec) for spec in REGISTRY],
                "feature_sets": FEATURE_SETS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _forbidden_input(name: str) -> bool:
    return name in FORBIDDEN_INPUTS or name.startswith(("mkt_", "market_", "odds_"))


def validate_registry() -> None:
    names = [spec.name for spec in REGISTRY]
    if len(names) != len(set(names)):
        raise ValueError("Registry contains duplicate feature names")
    for spec in REGISTRY:
        if spec.group not in GROUPS:
            raise ValueError(f"{spec.name}: unknown group {spec.group}")
        if spec.status not in STATUSES:
            raise ValueError(f"{spec.name}: unknown status {spec.status}")
        if spec.monotone_sign not in {-1, 0, 1}:
            raise ValueError(f"{spec.name}: invalid monotone sign")
        if any(_forbidden_input(value) for value in spec.inputs) and "lag" not in spec.leakage_note.lower():
            raise ValueError(f"{spec.name}: forbidden outcome input requires an explicit lag note")
    for set_name, features in FEATURE_SETS.items():
        unknown = set(features).difference(_BY_NAME)
        if unknown:
            raise ValueError(f"{set_name}: unknown features {sorted(unknown)}")
        inactive = [name for name in features if not _BY_NAME[name].implemented]
        if inactive:
            raise ValueError(f"{set_name}: unimplemented features {inactive}")


def _markdown() -> str:
    rows = [
        "| Name | Group | Models | Status | Implemented | Added in |",
        "|---|---|---|---|---|---|",
    ]
    for spec in REGISTRY:
        rows.append(
            f"| `{spec.name}` | {spec.group} | {', '.join(sorted(spec.models)) or '-'} | "
            f"{spec.status} | {spec.implemented} | {spec.added_in} |"
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--export", type=Path)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.list:
        print("\n".join(spec.name for spec in REGISTRY))
    elif args.export:
        export_json(args.export)
    elif args.validate:
        validate_registry()
        print(f"{REGISTRY_VERSION} {registry_hash()}")
    else:
        args.markdown.write_text(_markdown(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
