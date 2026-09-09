# Uncertainty-Aware Selective Autonomy for Robot Manipulation
### Learning when to **Act**, **Ask**, or **Defer**

A CUDA-free, Mac-friendly research pipeline that studies the *decision layer*:
given uncertainty about a tabletop manipulation task, should the robot execute
(**ACT**), ask a clarifying question (**ASK**), or hand off to a human (**DEFER**)?

The core pipeline deliberately **does not** run a physics engine or GPU
simulator. It models the world as ground-truth scene state plus principled
(simulated) uncertainty signals, so the whole contribution — cost-sensitive
selective autonomy, calibration, OOD behavior, the autonomy/safety frontier —
runs on a laptop CPU (or Apple MPS) in a couple of minutes. A pluggable backend
then swaps the abstract world for [ManiSkill](https://maniskill.readthedocs.io)
robot scenes without changing the decision layer.

---

## Research questions

- **RQ1** Can uncertainty-aware selective autonomy reduce task failures while
  requiring far fewer human interventions than an always-ask policy?
- **RQ2** Which uncertainty signals — language, perception, planning, safety, or
  OOD — matter most for deciding when to request help?
- **RQ3** Does calibrated uncertainty enable safer behavior under distribution
  shift?

## The decision problem

Each episode gives the policy a 10-dim feature vector (5 uncertainty signals +
scene statistics). It must choose `ACT`, `ASK`, or `DEFER`. Because this is a
simulator, an **oracle** knows the latent probabilities governing outcomes and
computes the cost-minimizing action as a training label:

```
a* = argmin_a  E[ C(a, world) ]

C(ACT)   = P(unsafe)·c_unsafe + (1-P(unsafe))·(1-P(correct))·c_wrong
C(ASK)   = c_ask   + P(unsafe)·c_unsafe      # clarifies identity, not danger
C(DEFER) = c_defer                            # human does it safely
```

Default costs: `c_unsafe=3.0`, `c_wrong=1.0`, `c_defer=0.3`, `c_ask=0.1`.
An unsafe collision is far worse than asking — so the policy learns to ask when
ambiguous and defer when dangerous.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No `pip install -e .` needed — every script adds the project root to the path
via `scripts/_bootstrap.py`.

## Run it

Quick end-to-end smoke test (tiny data, ~15 s):

```bash
python scripts/run_all.py --smoke
```

Full pipeline:

```bash
python scripts/run_all.py
```

Or run each stage on its own:

```bash
python scripts/generate_dataset.py     # -> results/data/*.csv
python scripts/train_policy.py          # -> results/models/*.pt
python scripts/evaluate.py              # main comparison table
python scripts/frontier.py              # autonomy/safety frontier plot
python scripts/calibration.py           # ECE + reliability diagram
python scripts/distribution_shift.py    # in-dist vs shifted
python scripts/ablation.py              # which signals matter
```

Every script accepts `--config configs/experiment.yaml` to override defaults.

Outputs land in `results/`: `tables/*.csv`, `plots/*.png`, `models/*.pt`,
`data/*.csv`.

---

## Policies (`uarsim/policies/`)

| Policy | Idea |
|---|---|
| `AlwaysAct` | always execute. Fast, zero queries, unsafe. |
| `AlwaysAsk` | always ask. Safe vs. ambiguity, huge effort. |
| `ThresholdPolicy` | weighted uncertainty + two thresholds. |
| `LearnedPolicy` | MLP classifier over the oracle optimal action. |
| `CostSensitivePolicy` | MLP that regresses expected costs, picks `argmin`. |

The **cost-sensitive** policy is the headline method: it reasons about the
*magnitude* of each mistake, not just a decision boundary.

## Repository layout

```
uarsim/                 # the library
├── config.py           # all defaults (costs, scene, shift, train)
├── types.py            # Action, Scene, Obj, Episode, ...
├── costs.py            # cost model + oracle optimal action + outcome sim
├── uncertainty.py      # the 5 observed uncertainty signals
├── data.py             # episode -> tabular row; feature schema
├── metrics.py          # evaluate a policy -> success/unsafe/effort/cost
├── nn.py               # tiny MLP + Mac (MPS/CPU) device selection
├── registry.py         # assemble the standard policy set
├── env/                # the abstract tabletop world
│   ├── objects.py      # vocabulary + geometry
│   ├── scene_generator.py
│   ├── task_generator.py   # instructions (ambiguity comes from here)
│   └── oracle.py       # privileged ground truth -> latents + labels
├── human.py            # simulated human (answers ASK / takes over on DEFER)
├── policies/           # ACT/ASK/DEFER policies
└── backends/           # pluggable scene sources
    ├── backend.py         # the Backend interface + EpisodeSample
    ├── synthetic_backend.py   # abstract scenes (no GPU)
    ├── maniskill_env.py       # ManiSkill custom scene (GPU)
    └── maniskill_backend.py   # real scene -> Scene/Obj mapping (GPU)
scripts/                # runnable experiments (see "Run it")
configs/                # optional YAML overrides
results/                # generated data, models, tables, plots
```

## What this is / isn't

**Is:** a clean study of risk-aware selective autonomy with an oracle,
cost-sensitive learning, calibration, OOD evaluation, and the
autonomy/safety trade-off.

**Isn't (yet):** real physics, perception, planning, LLMs, or closed-loop robot
control — the decision layer is built to plug straight into them via the scene
backends below.

## Scene backends — abstract or real robot

The decision layer is independent of *where* a scene comes from. A small backend
interface (`uarsim/backends/`) supplies scenes; everything downstream is reused:

- `SyntheticBackend` — generates abstract tabletop scenes; runs on your Mac (no GPU).
- `ManiSkillBackend` — reads a real ManiSkill tabletop scene (`TabletopChoice-v0`)
  and maps object poses/segmentation into the same `Scene`/`Obj` types, so
  `uncertainty.py`, `costs.py`, `policies/`, and `metrics.py` are reused verbatim.

```bash
python scripts/run_backend.py --backend synthetic --n 2000   # laptop, proves the flow
python scripts/run_backend.py --backend maniskill --n 500    # on a GPU box
```

The ManiSkill backend is written to the ManiSkill 3 API but authored on a Mac and
**not yet GPU-tested** — see [MANISKILL_SETUP.md](MANISKILL_SETUP.md) for the exact
SageMaker/Colab steps and the debugging loop.
