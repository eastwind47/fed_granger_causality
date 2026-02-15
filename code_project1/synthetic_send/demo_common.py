import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_config(config_path: Path = None) -> Dict:
    if config_path is None:
        config_path = Path(__file__).with_name("config.json")
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


def load_matrix(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def save_matrix(path: Path, mat: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    np.savetxt(path, np.asarray(mat, dtype=float), delimiter=",")


def load_vector(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    return np.asarray(arr, dtype=float).reshape(-1)


def save_vector(path: Path, vec: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    np.savetxt(path, np.asarray(vec, dtype=float).reshape(-1, 1), delimiter=",")


def normalize_y(y_raw: np.ndarray, d_m: int) -> np.ndarray:
    if y_raw.shape[0] == d_m:
        return y_raw
    if y_raw.shape[1] == d_m:
        return y_raw.T
    raise ValueError(f"Y shape {y_raw.shape} is incompatible with d_m={d_m}")


def load_component_data(
    data_dir: Path,
    client_id: int,
    eta_l: float,
    eta_g: float,
    lambda_l: float,
    force_zero_b: bool = True,
) -> Dict:
    comp_dir = Path(data_dir) / f"C{client_id}"
    a = load_matrix(comp_dir / "A.csv")
    c = load_matrix(comp_dir / "C.csv")

    y_raw = load_matrix(comp_dir / "Y.csv")
    d_m, p_m = c.shape
    y = normalize_y(y_raw, d_m)

    x0_path = comp_dir / "x0.csv"
    if x0_path.exists():
        x0 = load_matrix(x0_path).reshape(-1, 1)
    else:
        x0 = np.zeros((p_m, 1))

    q_path = comp_dir / "Q.csv"
    r_path = comp_dir / "R.csv"
    q = load_matrix(q_path) if q_path.exists() else 0.0005 * np.eye(p_m)
    r = load_matrix(r_path) if r_path.exists() else 0.0005 * np.eye(d_m)

    if force_zero_b:
        b = np.zeros((p_m, p_m))
    else:
        b_path = comp_dir / "B.csv"
        if b_path.exists():
            b_loaded = load_matrix(b_path)
            if b_loaded.shape == (p_m, p_m):
                b = b_loaded
            else:
                b = np.zeros((p_m, p_m))
        else:
            b = np.zeros((p_m, p_m))

    p0 = q.copy()
    return {
        "A": a,
        "B": b,
        "C": c,
        "Q": q,
        "R": r,
        "P0": p0,
        "x0": x0,
        "Y": y,
        "d_m": d_m,
        "p_m": p_m,
        "eta_l": float(eta_l),
        "eta_g": float(eta_g),
        "lambda_l": float(lambda_l),
    }


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metadata_path(state_dir: Path) -> Path:
    return Path(state_dir) / "state_metadata.json"


def client_state_dir(state_dir: Path, client_id: int) -> Path:
    return Path(state_dir) / f"client_{client_id}"


def server_state_dir(state_dir: Path) -> Path:
    return Path(state_dir) / "server"


def amn_path(state_dir: Path, m: int, n: int) -> Path:
    return server_state_dir(state_dir) / f"A_mn_{m}{n}.csv"


def write_gradient_message(path: Path, t: int, grad: np.ndarray) -> None:
    payload = np.concatenate(([float(t)], np.asarray(grad, dtype=float).reshape(-1)))
    save_vector(path, payload)


def read_gradient_message(path: Path, p_m: int) -> Tuple[int, np.ndarray]:
    payload = load_vector(path)
    expected_len = 1 + p_m
    if payload.size != expected_len:
        raise ValueError(
            f"Gradient message at {path} has length {payload.size}, expected {expected_len}"
        )
    t = int(round(payload[0]))
    grad = payload[1:].reshape(p_m, 1)
    return t, grad


def write_client_state_message(
    path: Path,
    t: int,
    x_dkf: np.ndarray,
    x_vfl: np.ndarray,
    x_vfl_pred: np.ndarray,
) -> None:
    payload = np.concatenate(
        (
            [float(t)],
            np.asarray(x_dkf, dtype=float).reshape(-1),
            np.asarray(x_vfl, dtype=float).reshape(-1),
            np.asarray(x_vfl_pred, dtype=float).reshape(-1),
        )
    )
    save_vector(path, payload)


def read_client_state_message(path: Path, p_m: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    payload = load_vector(path)
    expected_len = 1 + 3 * p_m
    if payload.size != expected_len:
        raise ValueError(
            f"Client state message at {path} has length {payload.size}, expected {expected_len}"
        )
    t = int(round(payload[0]))
    x_dkf = payload[1 : 1 + p_m].reshape(p_m, 1)
    x_vfl = payload[1 + p_m : 1 + 2 * p_m].reshape(p_m, 1)
    x_vfl_pred = payload[1 + 2 * p_m :].reshape(p_m, 1)
    return t, x_dkf, x_vfl, x_vfl_pred


def runtime_templates(cfg: Dict) -> Tuple[str, str]:
    runtime_cfg = cfg.get("runtime", {})
    grad_template = runtime_cfg.get("client_grad_file_template", "grad_client_{client_id}.csv")
    state_template = runtime_cfg.get("client_state_file_template", "state_client_{client_id}.csv")
    return grad_template, state_template

