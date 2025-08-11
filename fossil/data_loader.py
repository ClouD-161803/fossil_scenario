import csv
from typing import Sequence, Dict, List
import numpy as np
import torch

# ---------------------------------------------------------------------------
# CSV trajectory loading utilities
# ---------------------------------------------------------------------------
# Expected column layout (minimal): time,x1,x2,...,xN
# Optional derivatives: dx1,dx2,...,dxN (prefix d by default) OR custom prefix
# Optional trajectory grouping: an ID column whose name is provided via id_column
# If no ID column specified, each file is treated as a single trajectory.
# ---------------------------------------------------------------------------

def _finite_diff(times: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Compute simple finite difference derivatives for (n_vars, T) states."""
    if states.shape[1] < 2:
        # Not enough points, return zeros
        return np.zeros_like(states)
    dt = np.diff(times)
    dt[dt == 0] = 1e-8
    deriv = (states[:, 1:] - states[:, :-1]) / dt
    deriv = np.hstack([deriv, deriv[:, -1:]])
    return deriv

def load_csv_trajectories(
    paths: Sequence[str],
    n_vars: int,
    has_derivs: bool = False,
    id_column: str | None = None,
    deriv_prefix: str = "d",
) -> Dict[str, list]:
    """Load trajectories from one or more CSV files.

    Args:
        paths: sequence of CSV file paths.
        n_vars: number of state variables (expects columns x1..xN).
        has_derivs: whether derivative columns are present.
        id_column: optional column name to segment multiple trajectories.
        deriv_prefix: prefix for derivative columns (default 'd': dx1,...).

    Returns:
        dict with keys: times, states, derivs (each a list per trajectory)
    """
    traj_times: Dict[str, List[float]] = {}
    traj_states: Dict[str, List[List[float]]] = {}
    traj_derivs: Dict[str, List[List[float]]] = {}

    for path in paths:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if id_column and id_column not in cols:
                raise ValueError(f"ID column '{id_column}' not present in {path}")
            for row in reader:
                tid = row[id_column] if id_column else path  # group by file if no id
                t = float(row["time"]) if "time" in row else float(row[cols[0]])
                x = [float(row[f"x{i+1}"]) for i in range(n_vars)]
                if tid not in traj_times:
                    traj_times[tid] = []
                    traj_states[tid] = []
                    traj_derivs[tid] = []
                traj_times[tid].append(t)
                traj_states[tid].append(x)
                if has_derivs:
                    d = [float(row[f"{deriv_prefix}x{i+1}"]) for i in range(n_vars)]
                    traj_derivs[tid].append(d)

    times_list, states_list, derivs_list = [], [], []
    for tid in sorted(traj_times.keys()):
        times = np.array(traj_times[tid])
        order = np.argsort(times)
        times = times[order]
        states = np.array(traj_states[tid])[order].T  # (n_vars, T)
        if has_derivs and len(traj_derivs[tid]) == len(traj_states[tid]):
            derivs = np.array(traj_derivs[tid])[order].T
        else:
            derivs = _finite_diff(times, states)
        times_list.append(times)
        states_list.append(states)
        derivs_list.append(derivs)

    return {"times": times_list, "states": states_list, "derivs": derivs_list}


def to_training_data(traj_dict: Dict[str, list]) -> dict:
    """Convert trajectory dict into the internal DATA structure expected.

    Returns a dict with keys 'full_data' and 'states_only'. The latter is left
    empty for now (can be populated by user if domain boundary samples differ
    from trajectory samples).
    """
    return {
        "full_data": {
            "times": traj_dict["times"],
            "states": traj_dict["states"],
            "derivs": traj_dict["derivs"],
        },
        "states_only": {},
    }

__all__ = [
    "load_csv_trajectories",
    "to_training_data",
]
