"""Make the project root importable so scripts run without `pip install`.

Every script does `import _bootstrap` first. This lets you run e.g.
`python scripts/generate_dataset.py` from the project root with no setup.
"""
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ROOT = _ROOT
RESULTS = _ROOT / "results"
DATA_DIR = RESULTS / "data"
MODEL_DIR = RESULTS / "models"
TABLE_DIR = RESULTS / "tables"
PLOT_DIR = RESULTS / "plots"

for _d in (DATA_DIR, MODEL_DIR, TABLE_DIR, PLOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
