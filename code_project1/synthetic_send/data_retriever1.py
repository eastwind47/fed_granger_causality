import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_server_config(config_path: Optional[Path] = None) -> Dict:
    if config_path is None:
        config_path = Path(__file__).with_name("server_config.json")
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def resolve_path(cfg: Dict, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(cfg["_config_dir"]) / path).resolve()


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_matrix(path: Path, mat: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    np.savetxt(path, np.asarray(mat, dtype=float), delimiter=",")


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_matrix(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _load_optional_matrix(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return _load_matrix(path)


def _normalize_row_time(mat: np.ndarray, width: int, name: str) -> np.ndarray:
    """
    Return matrix with shape (T, width).
    Accepts either (T, width) or (width, T) and transposes if needed.
    """
    if mat.shape[1] == width:
        return mat
    if mat.shape[0] == width:
        return mat.T
    raise ValueError(
        f"{name} has shape {mat.shape}, expected second dimension {width} "
        f"or first dimension {width}."
    )


def _normalize_x0_col(x0_raw: np.ndarray, p_m: int) -> np.ndarray:
    flat = np.asarray(x0_raw, dtype=float).reshape(-1)
    if flat.size != p_m:
        raise ValueError(f"x0 has {flat.size} entries, expected {p_m}.")
    return flat.reshape(p_m, 1)


def _load_component_bundle(data_dir: Path, client_id: int) -> Dict:
    comp_dir = data_dir / f"C{client_id}"
    if not comp_dir.exists():
        raise FileNotFoundError(f"Missing component directory: {comp_dir}")

    a = _load_matrix(comp_dir / "A.csv")
    c = _load_matrix(comp_dir / "C.csv")
    d_m, p_m = c.shape

    y_row = _normalize_row_time(_load_matrix(comp_dir / "Y.csv"), d_m, f"C{client_id}/Y.csv")
    t_horizon = int(y_row.shape[0])

    q = _load_optional_matrix(comp_dir / "Q.csv")
    r = _load_optional_matrix(comp_dir / "R.csv")
    x0_raw = _load_optional_matrix(comp_dir / "x0.csv")

    if q is None:
        q = 0.0005 * np.eye(p_m)
    if r is None:
        r = 0.0005 * np.eye(d_m)
    if x0_raw is None:
        x0 = np.zeros((p_m, 1))
    else:
        x0 = _normalize_x0_col(x0_raw, p_m)

    x_dkf_pred_row = _load_optional_matrix(comp_dir / "X_dkf_pred.csv")
    x_dkf_est_row = _load_optional_matrix(comp_dir / "X_dkf_est.csv")
    if x_dkf_pred_row is not None:
        x_dkf_pred_row = _normalize_row_time(
            x_dkf_pred_row, p_m, f"C{client_id}/X_dkf_pred.csv"
        )
        if x_dkf_pred_row.shape[0] != t_horizon:
            raise ValueError(
                f"C{client_id}/X_dkf_pred.csv time length {x_dkf_pred_row.shape[0]} "
                f"does not match Y length {t_horizon}."
            )
    if x_dkf_est_row is not None:
        x_dkf_est_row = _normalize_row_time(
            x_dkf_est_row, p_m, f"C{client_id}/X_dkf_est.csv"
        )
        if x_dkf_est_row.shape[0] != t_horizon:
            raise ValueError(
                f"C{client_id}/X_dkf_est.csv time length {x_dkf_est_row.shape[0]} "
                f"does not match Y length {t_horizon}."
            )

    return {
        "client_id": client_id,
        "p_m": p_m,
        "d_m": d_m,
        "training_time": t_horizon,
        "A": a,
        "C": c,
        "Q": q,
        "R": r,
        "x0": x0,
        # LocalModel expects (d_m, T), so transpose row-per-time Y.
        "Y": y_row.T,
        "X_dkf_pred": None if x_dkf_pred_row is None else x_dkf_pred_row.T,
        "X_dkf_est": None if x_dkf_est_row is None else x_dkf_est_row.T,
    }


def build_startup_bundles(cfg: Dict) -> Tuple[Dict, Dict[int, Dict]]:
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    num_clients = int(cfg["system"]["num_clients"])
    if num_clients < 1:
        raise ValueError("system.num_clients must be >= 1.")

    gamma_g = float(cfg["global"]["gamma_g"])
    lambda_g = float(cfg["global"]["lambda_g"])

    p_dims = []
    d_dims = []
    training_time = None
    a_mm = {}
    client_bundles = {}

    for client_id in range(1, num_clients + 1):
        bundle = _load_component_bundle(data_dir, client_id)
        if training_time is None:
            training_time = int(bundle["training_time"])
        elif training_time != int(bundle["training_time"]):
            raise ValueError(
                f"Inconsistent time horizon: client_{client_id} has "
                f"T={bundle['training_time']}, expected T={training_time}."
            )

        p_dims.append(int(bundle["p_m"]))
        d_dims.append(int(bundle["d_m"]))
        a_mm[f"{client_id}{client_id}"] = bundle["A"]
        client_bundles[client_id] = bundle

    num_rounds = int(cfg.get("runtime", {}).get("num_rounds", training_time))
    if num_rounds < 1:
        raise ValueError("runtime.num_rounds must be >= 1.")
    if num_rounds > int(training_time):
        raise ValueError(
            f"runtime.num_rounds={num_rounds} exceeds training horizon T={training_time}."
        )

    init_cfg = cfg["initialization"]
    amn_mean = float(init_cfg["amn_mean"])
    amn_std = float(init_cfg["amn_std"])
    amn_seed = int(init_cfg["amn_seed"])
    rng = np.random.default_rng(amn_seed)

    a_mn = {}
    for m in range(1, num_clients + 1):
        for n in range(1, num_clients + 1):
            if m == n:
                continue
            a_mn[f"{m}{n}"] = rng.normal(amn_mean, amn_std, size=(p_dims[m - 1], p_dims[n - 1]))

    metadata = {
        "num_clients": num_clients,
        "training_time": int(training_time),
        "num_rounds": num_rounds,
        "p_dims": p_dims,
        "d_dims": d_dims,
        "data_dir": str(data_dir),
    }
    global_params = {"gamma_g": gamma_g, "lambda_g": lambda_g}

    server_bundle = {
        "metadata": metadata,
        "global": global_params,
        "A_mm": a_mm,
        "A_mn": a_mn,
    }
    return server_bundle, client_bundles


def persist_server_bundle(cfg: Dict, server_bundle: Dict) -> None:
    runtime_cfg = cfg.get("runtime", {})
    if not bool(runtime_cfg.get("persist_server_bundle", True)):
        return

    state_dir = resolve_path(cfg, cfg["paths"]["server_state_dir"])
    ensure_dir(state_dir)
    save_json(state_dir / "server_metadata.json", server_bundle["metadata"])
    save_json(state_dir / "global_params.json", server_bundle["global"])

    for key, block in server_bundle["A_mm"].items():
        save_matrix(state_dir / f"A_mm_{key}.csv", block)
    for key, block in server_bundle["A_mn"].items():
        save_matrix(state_dir / f"A_mn_{key}.csv", block)
