"""Pluggable scene backends behind one small interface.

The decision layer (uncertainty signals, oracle cost model, policies, metrics)
is independent of *where* a scene comes from. A backend supplies scenes; the
rest of the codebase is reused unchanged.

Two backends implement :class:`Backend`:

  * ``SyntheticBackend`` — generates abstract tabletop scenes in pure Python.
                           Runs anywhere (CPU / Apple MPS), no GPU. This is the
                           reference every other backend must match.
  * ``ManiSkillBackend``  — reads a real ManiSkill tabletop scene (object poses,
                           camera segmentation) and maps it into the same
                           ``Scene``/``Obj`` types. Needs a CUDA GPU and the
                           optional ManiSkill dependency — see MANISKILL_SETUP.md.

``scripts/run_backend.py`` works with either, so the whole flow can run on a
laptop with ``--backend synthetic`` and switch to ``--backend maniskill`` on a
GPU with no other change.
"""
