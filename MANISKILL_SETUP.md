# Running the ManiSkill backend on a GPU (SageMaker / Colab / RunPod)

The ManiSkill backend swaps the abstract world for a **real ManiSkill 3 robot
scene**. The ACT/ASK/DEFER decision layer is unchanged — only the *scene source*
differs.

> **You do not need any of this for day-to-day work.** The `synthetic` backend
> runs on your Mac. Only the `maniskill` backend needs a GPU.

---

## 0. First, prove the flow on your Mac (no GPU)

```bash
python scripts/train_policy.py          # if you haven't already
python scripts/run_backend.py --backend synthetic --n 2000
```

If that prints a results table, the whole backend flow works. On the GPU you
only change `--backend synthetic` to `--backend maniskill`.

---

## 1. Move the code to the GPU box (the GitHub bridge)

On your Mac:
```bash
cd "Uncertainty Aware Robot Simulation"
git init && git add -A && git commit -m "simulator + maniskill backend"
# create an empty repo on github.com, then:
git remote add origin https://github.com/<you>/uncertainty-aware-robot.git
git push -u origin main
```

On the GPU box (SageMaker terminal / Colab cell):
```bash
git clone https://github.com/<you>/uncertainty-aware-robot.git
cd uncertainty-aware-robot
```

Whenever the code changes, `git pull` on the GPU box and re-run.

---

## 2A. AWS SageMaker Studio Lab (free, Colab-like)

1. Sign up at studiolab.sagemaker.aws (email only, no AWS account).
2. Start a runtime with **GPU** and open it (JupyterLab).
3. Open a **Terminal** (File → New → Terminal) and run:
   ```bash
   git clone <your-repo-url> && cd uncertainty-aware-robot
   pip install -r requirements.txt
   pip install -r requirements-maniskill.txt
   python -c "import mani_skill; print('maniskill', mani_skill.__version__)"
   ```
4. Sanity-check ManiSkill itself (its own demo) before ours:
   ```bash
   python -m mani_skill.examples.demo_random_action -e PickCube-v1
   ```
   If that errors, it's a ManiSkill/GPU setup issue (usually Vulkan) — paste
   the error back to Claude and we fix it before touching our code.
5. Run the backend:
   ```bash
   python scripts/train_policy.py                  # or git-pull the .pt models
   python scripts/run_backend.py --backend maniskill --n 500 --num-cubes 3
   python scripts/run_backend.py --backend maniskill --n 500 --mode shift
   ```

## 2B. Google Colab (often the smoothest)

In a notebook:
```python
!git clone <your-repo-url>
%cd uncertainty-aware-robot
!pip install -r requirements.txt -q
!pip install -r requirements-maniskill.txt -q
!python -m mani_skill.examples.demo_random_action -e PickCube-v1   # smoke test
!python scripts/run_backend.py --backend maniskill --n 300
```
Set **Runtime → Change runtime type → GPU** first.

---

## 3. What to expect (and the honest caveats)

- `uarsim/backends/maniskill_env.py` and `maniskill_backend.py` are written to
  the ManiSkill 3 API but authored on a Mac and **not yet GPU-tested**. The lines
  most likely to need a tweak are marked `# MANISKILL:`.
- **The debugging loop:** run a command → copy the full traceback → paste it to
  Claude → we edit → you `git pull` and re-run. Two or three rounds is normal
  for a first bring-up.
- The results (success / unsafe / human-effort) should tell the same story as
  the synthetic backend, now on real rendered scenes. Sample frames are saved to
  `results/plots/backend_frames/`.

## 4. Optional extensions

- **Closed-loop execution:** actually grasp on ACT using ManiSkill's Panda
  motion-planning solutions, and read the real success flag instead of the
  privileged outcome. (`pip install mplib`; see ManiSkill's
  `examples/motionplanning`.)
- **Real perception uncertainty:** replace the segmentation-based confidence
  proxy with a small trained detector's score.
- **Real OOD:** add unseen object shapes/textures for the `--mode shift` runs.
