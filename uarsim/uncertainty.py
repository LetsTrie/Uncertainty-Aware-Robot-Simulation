"""Derive the five observed uncertainty signals for an episode.

This module does not run real perception / language / planning models. Instead we
compute each signal from ground-truth scene structure and then corrupt it with
noise, so the learned policy sees a realistic *noisy estimate* rather than the
latent truth. Every signal is normalized to [0, 1]:

    U_language   : referential ambiguity  (how many objects match the command)
    U_perception : object-identification uncertainty
    U_planning   : motion / reachability difficulty
    U_ood        : unfamiliarity of the scene / objects
    U_safety     : predicted risk of an unsafe collision

Staged design: these are simulated now and can later be replaced by learned
estimators without changing anything downstream.
"""
from __future__ import annotations

import numpy as np

SIGNALS = ["u_language", "u_perception", "u_planning", "u_ood", "u_safety"]


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def compute_signals(scene, target, n_matches, latents, cfg, rng) -> dict:
    """Return the noisy, observed uncertainty signals for one episode.

    ``cfg`` is a SceneConfig/ShiftConfig (supplies noise levels).
    """
    n = len(scene.objects)
    frac_novel = sum(o.novel for o in scene.objects) / max(n, 1)

    # Language ambiguity: 0 when exactly one object matches, ->1 as matches grow.
    u_language = 1.0 - 1.0 / max(n_matches, 1)

    # Perception: complement of the target's identification confidence.
    u_perception = 1.0 - target.perception_conf

    # Planning: farther + more cluttered targets are harder to reach.
    dist = latents["target_distance"]
    u_planning = 0.55 * scene.clutter + 0.45 * dist

    # Out-of-distribution: dominated by whether the *target* is novel.
    u_ood = 0.7 * float(target.novel) + 0.3 * frac_novel

    # Safety: a (deliberately imperfect) estimate of the true collision risk.
    u_safety = latents["p_unsafe"]

    n_noise = cfg.sig_noise
    s_noise = cfg.safety_noise
    signals = {
        "u_language": _clip01(u_language + rng.normal(0, n_noise * 0.5)),
        "u_perception": _clip01(u_perception + rng.normal(0, n_noise)),
        "u_planning": _clip01(u_planning + rng.normal(0, n_noise)),
        "u_ood": _clip01(u_ood + rng.normal(0, n_noise)),
        # Safety sensing is noisier — and gets *less reliable* under shift,
        # which is the crux of the calibration behavior under distribution shift.
        "u_safety": _clip01(u_safety + rng.normal(0, s_noise)),
    }
    return signals
