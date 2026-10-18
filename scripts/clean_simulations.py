"""
Delete simulated missions (sim_*) from the database and data folders.

    python scripts/clean_simulations.py              # keep the 5 most recent
    python scripts/clean_simulations.py --keep 0     # delete all of them
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.simulation.service import delete_simulation, list_simulations  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=5, help="most recent missions to keep")
    args = ap.parse_args()
    sims = list_simulations()                       # ids sort by creation time
    doomed = sims[:-args.keep] if args.keep > 0 else sims
    for sim_id in doomed:
        delete_simulation(sim_id)
    print(f"Deleted {len(doomed)} simulated mission(s); kept {len(sims) - len(doomed)}.")


if __name__ == "__main__":
    main()
