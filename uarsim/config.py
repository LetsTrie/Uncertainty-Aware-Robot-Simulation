"""Configuration dataclasses with sane defaults.

Everything runs from these defaults with zero setup. YAML files in
``configs/`` can override fields via :func:`load_config`, but are optional.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
import copy


@dataclass
class CostConfig:
    """Cost model. Costs are positive; a successful task costs 0."""

    c_wrong: float = 1.0    # acted on the wrong object
    c_unsafe: float = 3.0   # unsafe collision (the failure we most want to avoid)
    c_ask: float = 0.1      # one clarification question
    c_defer: float = 0.3    # hand the whole task to a human


@dataclass
class SceneConfig:
    min_objects: int = 2
    max_objects: int = 6

    # Probabilities that shape instruction ambiguity.
    p_full_ref: float = 0.55     # "the blue mug" (color + category)
    p_cat_only: float = 0.30     # "the mug"
    p_generic: float = 0.15      # "the object"

    # Base object property rates (train distribution).
    p_unreachable: float = 0.08
    p_fragile: float = 0.18
    novel_prob: float = 0.0      # train distribution has no novel objects

    # Noise on observed uncertainty signals (std of Gaussian, pre-clip).
    sig_noise: float = 0.05
    safety_noise: float = 0.06


@dataclass
class ShiftConfig(SceneConfig):
    """Out-of-distribution test scenes: novel objects, more clutter, more
    fragility, more ambiguity, and noisier/less-reliable safety sensing."""

    min_objects: int = 3
    max_objects: int = 8
    p_full_ref: float = 0.40
    p_cat_only: float = 0.33
    p_generic: float = 0.27
    p_unreachable: float = 0.16
    p_fragile: float = 0.34
    novel_prob: float = 0.45
    sig_noise: float = 0.09
    safety_noise: float = 0.14


@dataclass
class DataConfig:
    seed: int = 0
    n_train: int = 40_000
    n_val: int = 8_000
    n_test_id: int = 20_000    # in-distribution test
    n_test_shift: int = 20_000  # shifted test


@dataclass
class ThresholdConfig:
    """Fixed-threshold uncertainty baseline."""

    # Weights for the combined uncertainty U = sum(w_i * U_i).
    w_language: float = 1.0
    w_perception: float = 1.0
    w_planning: float = 1.0
    w_ood: float = 1.0
    w_safety: float = 2.0   # safety weighted higher
    t_act: float = 0.30     # U < t_act        -> ACT
    t_defer: float = 0.70   # U >= t_defer      -> DEFER (ASK in between)


@dataclass
class TrainConfig:
    hidden: tuple = (32, 64, 32)
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0


@dataclass
class Config:
    cost: CostConfig = None
    scene: SceneConfig = None
    shift: ShiftConfig = None
    data: DataConfig = None
    threshold: ThresholdConfig = None
    train: TrainConfig = None

    def __post_init__(self):
        self.cost = self.cost or CostConfig()
        self.scene = self.scene or SceneConfig()
        self.shift = self.shift or ShiftConfig()
        self.data = self.data or DataConfig()
        self.threshold = self.threshold or ThresholdConfig()
        self.train = self.train or TrainConfig()


def _apply_overrides(dc, overrides: dict):
    """Set matching dataclass fields from a plain dict (ignores unknown keys)."""
    valid = {f.name for f in fields(dc)}
    for k, v in (overrides or {}).items():
        if k in valid:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None = None) -> Config:
    """Load defaults, optionally overriding from a YAML file.

    The YAML may contain top-level sections ``cost``, ``scene``, ``shift``,
    ``data``, ``threshold``, ``train`` with any subset of fields.
    """
    cfg = Config()
    if path is None:
        return cfg
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "PyYAML is required to load a --config file. Run without --config "
            "to use defaults, or `pip install pyyaml`."
        ) from e
    data = yaml.safe_load(Path(path).read_text()) or {}
    _apply_overrides(cfg.cost, data.get("cost"))
    _apply_overrides(cfg.scene, data.get("scene"))
    _apply_overrides(cfg.shift, data.get("shift"))
    _apply_overrides(cfg.data, data.get("data"))
    _apply_overrides(cfg.threshold, data.get("threshold"))
    _apply_overrides(cfg.train, data.get("train"))
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return {
        "cost": asdict(cfg.cost),
        "scene": asdict(cfg.scene),
        "shift": asdict(cfg.shift),
        "data": asdict(cfg.data),
        "threshold": asdict(cfg.threshold),
        "train": asdict(cfg.train),
    }
