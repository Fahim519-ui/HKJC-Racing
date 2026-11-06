"""Registry-only feature exclusions for fold experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

try:
    from model_logic.feature_registry import TRAINER_RATE_FEATURES
except ModuleNotFoundError:
    from feature_registry import TRAINER_RATE_FEATURES

# Trainer-rate ablations drop every feature built from trainer win rates, including the
# v3.1 jockey/trainer strength x field size interaction.
_TRAINER = tuple(sorted(TRAINER_RATE_FEATURES))


@dataclass(frozen=True)
class AblationConfig:
    name: str
    exclude_names: tuple[str, ...] = ()
    exclude_groups: tuple[str, ...] = ()
    primary: bool = False

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CONFIGS = {
    "primary": AblationConfig("primary", primary=True),
    "no_trainer_rate": AblationConfig(
        "no_trainer_rate", exclude_names=_TRAINER
    ),
    "no_track_distance": AblationConfig(
        "no_track_distance", exclude_groups=("track_distance",)
    ),
    "no_trainer_no_track_distance": AblationConfig(
        "no_trainer_no_track_distance",
        exclude_names=_TRAINER,
        exclude_groups=("track_distance",),
    ),
}

if sum(config.primary for config in CONFIGS.values()) != 1:
    raise RuntimeError("Exactly one ablation config must be primary")


def get_config(name: str) -> AblationConfig:
    try:
        return CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"Unknown ablation config: {name}") from error
