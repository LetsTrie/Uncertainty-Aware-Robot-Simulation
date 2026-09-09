"""Run the whole pipeline end to end.

  python scripts/run_all.py            # full run
  python scripts/run_all.py --smoke    # tiny/fast run to check everything works

Runs: generate -> train -> evaluate -> frontier -> calibration ->
distribution_shift -> ablation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import _bootstrap  # noqa: F401


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=_bootstrap.ROOT)
    print(f"  ...done in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny dataset + few epochs, just to verify the pipeline")
    args = ap.parse_args()
    py = sys.executable
    s = "scripts"

    gen = [py, f"{s}/generate_dataset.py"]
    train = [py, f"{s}/train_policy.py"]
    abl = [py, f"{s}/ablation.py"]
    if args.smoke:
        gen += ["--n-train", "4000", "--n-val", "1000",
                "--n-test-id", "2000", "--n-test-shift", "2000"]
        train += ["--epochs", "8"]
        abl += ["--epochs", "8"]

    run(gen)
    run(train)
    run([py, f"{s}/evaluate.py"])
    run([py, f"{s}/frontier.py"])
    run([py, f"{s}/calibration.py"])
    run([py, f"{s}/distribution_shift.py"])
    run(abl)
    print("\nAll done. See results/tables/ and results/plots/.")


if __name__ == "__main__":
    main()
